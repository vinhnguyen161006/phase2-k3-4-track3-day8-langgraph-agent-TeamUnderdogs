"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.

All four functions are pure and side-effect free: they only *read* state. That
keeps them unit-testable without an LLM or a compiled graph, and it means a
checkpoint replay always takes the same branch for the same state.
"""

from __future__ import annotations

from .state import AgentState, Route

#: classify_node's route → the node that handles it.
_ROUTE_TO_NODE: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}

#: Where an unrecognised route lands. "answer" degrades gracefully — the LLM
#: still produces a grounded response instead of the graph dead-ending.
_DEFAULT_NODE = "answer"


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node.

    Unknown or empty routes fall back to ``answer`` so a malformed LLM
    classification degrades to a plain response instead of hanging the graph.
    """
    return _ROUTE_TO_NODE.get(state.get("route", ""), _DEFAULT_NODE)


def route_after_evaluate(state: AgentState) -> str:
    """Decide if the tool result is satisfactory or needs a retry.

    This is the 'done?' check that closes the retry loop — the key LangGraph
    advantage over a linear LCEL chain, which can only run forward.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or escalate to the dead-letter node.

    Bounded by construction: ``retry_or_fallback_node`` increments ``attempt``
    before this runs, so each pass through the loop strictly increases the
    counter and the ``attempt >= max_attempts`` branch is always reachable.
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on the human approval decision.

    Approved risky actions proceed to the tool; rejected ones go to ``clarify``
    so the user is asked for an alternative rather than silently dropped.
    """
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"
