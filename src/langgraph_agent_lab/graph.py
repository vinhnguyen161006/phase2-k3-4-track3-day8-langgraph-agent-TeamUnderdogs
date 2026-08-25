"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.

Topology
--------
::

    START -> intake -> classify -> [route_after_classify]
      simple       -> answer -> finalize -> END
      tool         -> tool -> evaluate -> [route_after_evaluate]
                                            success     -> answer -> finalize -> END
                                            needs_retry -> retry -> [route_after_retry]
                                                    attempt <  max -> tool (loop)
                                                    attempt >= max -> dead_letter -> finalize -> END
      missing_info -> clarify -> finalize -> END
      risky        -> risky_action -> approval -> [route_after_approval]
                                                    approved -> tool -> evaluate -> ...
                                                    rejected -> clarify -> finalize -> END
      error        -> retry -> [route_after_retry] -> ...

Every terminal path converges on ``finalize`` before ``END``, which is what
makes "did this run finish?" a single check on the event trail.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph workflow.

    Args:
        checkpointer: optional LangGraph checkpointer. When supplied, each run
            must pass ``config={"configurable": {"thread_id": ...}}`` so state is
            persisted per scenario and can be inspected or resumed later.

    Returns:
        The compiled graph, ready for ``.invoke()`` / ``.stream()``.
    """
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    builder = StateGraph(AgentState)

    # ── 1. register nodes (11 total) ────────────────────────────────
    builder.add_node("intake", intake_node)
    builder.add_node("classify", classify_node)
    builder.add_node("tool", tool_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("answer", answer_node)
    builder.add_node("clarify", ask_clarification_node)
    builder.add_node("risky_action", risky_action_node)
    builder.add_node("approval", approval_node)
    builder.add_node("retry", retry_or_fallback_node)
    builder.add_node("dead_letter", dead_letter_node)
    builder.add_node("finalize", finalize_node)

    # ── 2. fixed edges ──────────────────────────────────────────────
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("tool", "evaluate")
    builder.add_edge("risky_action", "approval")
    builder.add_edge("answer", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("dead_letter", "finalize")
    builder.add_edge("finalize", END)

    # ── 3. conditional edges ────────────────────────────────────────
    # The explicit path maps double as documentation and let LangGraph draw an
    # accurate diagram; a router returning anything outside these keys errors
    # loudly rather than silently stalling.
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"answer": "answer", "retry": "retry"},
    )
    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    return builder.compile(checkpointer=checkpointer)


def render_mermaid(checkpointer: Any | None = None) -> str:
    """Return the compiled graph as a Mermaid diagram (documentation extension)."""
    return build_graph(checkpointer=checkpointer).get_graph().draw_mermaid()
