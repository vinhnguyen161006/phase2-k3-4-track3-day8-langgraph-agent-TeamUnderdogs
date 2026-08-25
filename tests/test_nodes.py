"""Node-level tests that run offline (no API key required).

LLM-backed nodes are exercised through a stubbed ``get_llm`` so the deterministic
logic around the model call — retry simulation, approval gating, the evaluate
gate, dead-letter escalation — is covered in CI without network access.
"""

from __future__ import annotations

import pytest

from langgraph_agent_lab import nodes
from langgraph_agent_lab.state import Route, make_event


# ─── tool_node: failure simulation drives the retry loop ─────────────
def test_tool_node_fails_early_on_error_route():
    result = nodes.tool_node({"route": Route.ERROR.value, "attempt": 0, "query": "timeout"})
    assert result["tool_results"][0].startswith("ERROR")


def test_tool_node_succeeds_after_two_attempts():
    result = nodes.tool_node({"route": Route.ERROR.value, "attempt": 2, "query": "timeout"})
    assert result["tool_results"][0].startswith("OK")


def test_tool_node_succeeds_immediately_on_tool_route():
    result = nodes.tool_node({"route": Route.TOOL.value, "attempt": 0, "query": "lookup order 1"})
    assert result["tool_results"][0].startswith("OK")


def test_tool_node_marks_approved_action_as_executed():
    result = nodes.tool_node(
        {
            "route": Route.RISKY.value,
            "attempt": 0,
            "query": "refund customer",
            "approval": {"approved": True},
        }
    )
    assert "approved action executed" in result["tool_results"][0]


# ─── evaluate_node: the retry gate ───────────────────────────────────
def test_evaluate_node_flags_error_without_calling_llm(monkeypatch):
    def boom(*args, **kwargs):  # would fail if the judge were called
        raise AssertionError("LLM must not be called for an explicit ERROR marker")

    monkeypatch.setattr(nodes, "get_llm", boom)
    result = nodes.evaluate_node({"tool_results": ["ERROR: service unavailable"]})
    assert result["evaluation_result"] == "needs_retry"


def test_evaluate_node_needs_retry_when_no_tool_result():
    result = nodes.evaluate_node({"tool_results": []})
    assert result["evaluation_result"] == "needs_retry"


def test_evaluate_node_accepts_result_when_judge_unavailable(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    result = nodes.evaluate_node({"tool_results": ["OK: found order"], "query": "q"})
    assert result["evaluation_result"] == "success"


# ─── retry / dead letter ─────────────────────────────────────────────
def test_retry_node_increments_attempt():
    result = nodes.retry_or_fallback_node(
        {"attempt": 1, "max_attempts": 3, "tool_results": ["ERROR"]}
    )
    assert result["attempt"] == 2
    assert result["errors"]


def test_dead_letter_sets_final_answer():
    result = nodes.dead_letter_node({"attempt": 3, "max_attempts": 3, "query": "broken"})
    assert result["final_answer"]
    assert "escalated" in result["final_answer"].lower()


# ─── approval / HITL ─────────────────────────────────────────────────
def test_approval_node_mock_approves_by_default(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_INTERRUPT", raising=False)
    result = nodes.approval_node({"proposed_action": "refund", "risk_level": "high"})
    assert result["approval"]["approved"] is True
    assert result["events"][0]["node"] == "approval"


def test_risky_action_node_records_proposal():
    result = nodes.risky_action_node({"query": "delete account", "risk_level": "high"})
    assert "delete account" in result["proposed_action"]
    assert result["events"][0]["event_type"] == "pending_approval"


# ─── finalize ────────────────────────────────────────────────────────
def test_finalize_emits_completed_event():
    result = nodes.finalize_node({"route": "simple", "final_answer": "hi"})
    event = result["events"][0]
    assert event["node"] == "finalize"
    assert event["metadata"]["answered"] is True


# ─── LLM nodes degrade instead of raising ────────────────────────────
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Refund this customer and send confirmation email", Route.RISKY.value),
        ("Please lookup order status for order 12345", Route.TOOL.value),
        ("Timeout failure while processing request", Route.ERROR.value),
        ("Can you fix it?", Route.MISSING_INFO.value),
    ],
)
def test_classify_node_falls_back_when_llm_unavailable(monkeypatch, query, expected):
    monkeypatch.setattr(
        nodes, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    result = nodes.classify_node({"query": query})
    assert result["route"] == expected
    assert result["errors"]  # the failure is recorded, not swallowed


def test_answer_node_falls_back_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    result = nodes.answer_node({"query": "where is my order", "tool_results": ["OK: in transit"]})
    assert result["final_answer"]
    assert result["errors"]


def test_classify_node_uses_llm_when_available(monkeypatch):
    """The LLM verdict wins over the heuristic — proves the call is real."""

    class _Stub:
        def with_structured_output(self, schema):
            return self

        def invoke(self, _messages):
            return nodes.Classification(route="risky", risk_level="high", reason="stubbed")

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _Stub())
    # Wording a heuristic would call "simple"; the stubbed LLM says risky.
    result = nodes.classify_node({"query": "what are your opening hours"})
    assert result["route"] == "risky"
    assert result["events"][0]["metadata"]["source"] == "llm"
    assert "errors" not in result


# ─── event helper ────────────────────────────────────────────────────
def test_make_event_promotes_latency_to_typed_field():
    event = make_event("answer", "completed", "msg", latency_ms=123, source="llm")
    assert event["latency_ms"] == 123
    assert "latency_ms" not in event["metadata"]
    assert event["metadata"]["source"] == "llm"


# ─── rate-limit retry helper ─────────────────────────────────────────
def test_call_with_retry_retries_rate_limit_then_succeeds(monkeypatch):
    from langgraph_agent_lab import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)  # no real backoff
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return "ok"

    assert llm_mod.call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_call_with_retry_does_not_retry_other_errors(monkeypatch):
    from langgraph_agent_lab import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad schema")

    with pytest.raises(ValueError):
        llm_mod.call_with_retry(broken)
    assert calls["n"] == 1  # non-rate-limit errors surface immediately
