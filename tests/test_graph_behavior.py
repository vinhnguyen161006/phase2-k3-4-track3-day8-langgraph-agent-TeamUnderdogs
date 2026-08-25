"""End-to-end graph behavior tests that run offline.

``classify_node`` is stubbed so each test pins one route deterministically; the
rest of the graph (retry loop, HITL gate, dead-letter escalation, termination)
runs for real. This covers graph *behavior* in CI without an API key —
``test_graph_smoke.py`` covers the same paths with a live LLM.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
)

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    """Force every LLM-backed node onto its deterministic fallback path."""
    monkeypatch.setattr(
        nodes, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline test"))
    )


def _pin_route(monkeypatch, route: str) -> None:
    """Pin classify_node's verdict so the test targets one branch."""
    real = nodes.classify_node

    def stub(state):
        result = dict(real(state))
        result["route"] = route
        result["risk_level"] = "high" if route == Route.RISKY.value else "low"
        result.pop("errors", None)
        return result

    monkeypatch.setattr(nodes, "classify_node", stub)


def _run(scenario: Scenario):
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    state = initial_state(scenario)
    return graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})


def _nodes_visited(result) -> list[str]:
    return [event["node"] for event in result.get("events", [])]


@pytest.mark.parametrize(
    "route",
    [
        Route.SIMPLE.value,
        Route.TOOL.value,
        Route.MISSING_INFO.value,
        Route.RISKY.value,
        Route.ERROR.value,
    ],
)
def test_every_route_terminates_at_finalize(monkeypatch, route):
    _pin_route(monkeypatch, route)
    result = _run(Scenario(id=f"t-{route}", query="a test ticket", expected_route=Route(route)))
    assert "finalize" in _nodes_visited(result)
    assert result.get("final_answer") or result.get("pending_question")


def test_error_route_retries_then_succeeds(monkeypatch):
    """Default budget: tool fails while attempt < 2, so the loop runs twice."""
    _pin_route(monkeypatch, Route.ERROR.value)
    result = _run(
        Scenario(id="retry", query="timeout failure", expected_route=Route.ERROR, max_attempts=3)
    )
    visited = _nodes_visited(result)
    assert visited.count("retry") == 2
    assert result["attempt"] == 2
    assert "dead_letter" not in visited
    assert result["final_answer"]


def test_retry_loop_is_bounded_and_escalates(monkeypatch):
    """max_attempts=1 can never reach the success branch -> dead letter."""
    _pin_route(monkeypatch, Route.ERROR.value)
    result = _run(
        Scenario(
            id="dead", query="system failure cannot recover",
            expected_route=Route.ERROR, max_attempts=1,
        )
    )
    visited = _nodes_visited(result)
    assert "dead_letter" in visited
    assert result["attempt"] == 1
    assert "escalated" in result["final_answer"].lower()
    assert visited[-1] == "finalize"


def test_risky_route_requires_approval_before_tool(monkeypatch):
    """The HITL gate: approval must precede any tool execution."""
    _pin_route(monkeypatch, Route.RISKY.value)
    result = _run(
        Scenario(
            id="risky", query="refund this customer",
            expected_route=Route.RISKY, requires_approval=True,
        )
    )
    visited = _nodes_visited(result)
    assert "risky_action" in visited and "approval" in visited
    assert visited.index("approval") < visited.index("tool")
    assert result["approval"]["approved"] is True
    assert result["proposed_action"]


def test_rejected_approval_routes_to_clarify_not_tool(monkeypatch):
    """A declined risky action must never execute."""
    _pin_route(monkeypatch, Route.RISKY.value)
    monkeypatch.setattr(
        nodes,
        "approval_node",
        lambda state: {
            "approval": {"approved": False, "reviewer": "human", "comment": "denied"},
            "events": [
                {
                    "node": "approval", "event_type": "rejected", "message": "denied",
                    "latency_ms": 0, "metadata": {},
                }
            ],
        },
    )
    result = _run(
        Scenario(id="rejected", query="delete the account", expected_route=Route.RISKY)
    )
    visited = _nodes_visited(result)
    assert "tool" not in visited, "rejected action must not reach the tool"
    assert "clarify" in visited
    assert result.get("pending_question")


def test_simple_route_skips_tool_entirely(monkeypatch):
    _pin_route(monkeypatch, Route.SIMPLE.value)
    result = _run(
        Scenario(id="simple", query="how do I reset my password", expected_route=Route.SIMPLE)
    )
    visited = _nodes_visited(result)
    assert visited == ["intake", "classify", "answer", "finalize"]


def test_append_only_reducers_accumulate_across_retries(monkeypatch):
    """Audit trails must keep every attempt, not just the last one."""
    _pin_route(monkeypatch, Route.ERROR.value)
    result = _run(
        Scenario(id="reducers", query="timeout", expected_route=Route.ERROR, max_attempts=3)
    )
    # The error route enters `retry` first (attempt -> 1), so `tool` runs at
    # attempt=1 (fails) and attempt=2 (succeeds): two accumulated results.
    assert len(result["tool_results"]) == 2
    assert result["tool_results"][0].startswith("ERROR")
    assert result["tool_results"][-1].startswith("OK")
    assert len(result["errors"]) >= 2  # one per retry, never overwritten
    assert len(result["events"]) >= 8


def test_checkpointer_persists_state_history(monkeypatch):
    """thread_id per run + walkable super-step history (persistence evidence)."""
    _pin_route(monkeypatch, Route.SIMPLE.value)
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="persist", query="hello there", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=config)

    snapshot = graph.get_state(config)
    assert snapshot.values["final_answer"]
    assert len(list(graph.get_state_history(config))) > 1
