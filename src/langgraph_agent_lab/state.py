"""State schema for the Day 08 LangGraph lab.

Design rule: keep state lean and JSON-serializable so every field survives a
checkpoint round-trip (SQLite/Postgres store plain JSON, not Python objects).

Reducer policy
--------------
Two kinds of fields live here:

* **Append-only** (``Annotated[list[...], add]``) — anything that forms an audit
  trail. ``messages``, ``tool_results``, ``errors`` and ``events`` must never
  lose history, because the retry loop visits the same nodes several times and
  metrics are derived by counting those entries.
* **Overwrite** (plain annotation) — anything that describes the *current*
  situation only: ``route``, ``attempt``, ``evaluation_result``, ``approval``…
  Keeping these as last-write-wins is what lets the retry loop re-enter
  ``tool``/``evaluate`` without accumulating stale verdicts.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


#: Routes an LLM is allowed to emit from ``classify_node``.
CLASSIFIABLE_ROUTES: tuple[str, ...] = (
    Route.RISKY.value,
    Route.TOOL.value,
    Route.MISSING_INFO.value,
    Route.ERROR.value,
    Route.SIMPLE.value,
)


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Fields without an ``Annotated[..., add]`` reducer are overwrite-on-write;
    the four list fields at the bottom are append-only audit trails.
    """

    # ── identity / input (overwrite) ────────────────────────────────
    thread_id: str
    scenario_id: str
    query: str

    # ── classification result (overwrite) ───────────────────────────
    route: str
    risk_level: str
    classify_reason: str

    # ── retry-loop bookkeeping (overwrite) ──────────────────────────
    attempt: int
    max_attempts: int
    #: "success" | "needs_retry" — gate read by ``route_after_evaluate``.
    evaluation_result: str

    # ── clarification flow (overwrite) ──────────────────────────────
    #: Question asked back to the user when the query is not actionable.
    pending_question: str | None

    # ── risky-action / HITL flow (overwrite) ────────────────────────
    #: Human-readable description of the side-effecting action awaiting sign-off.
    proposed_action: str | None
    #: Serialized ``ApprovalDecision`` — dict, not the model, so it checkpoints.
    approval: dict[str, Any] | None

    # ── output (overwrite) ──────────────────────────────────────────
    final_answer: str | None

    # ── audit trails (append-only) ──────────────────────────────────
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "classify_reason": "",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "evaluation_result": "",
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "final_answer": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload.

    ``latency_ms`` is promoted out of the metadata bag into the typed field so
    metrics can sum it without reaching into free-form metadata.
    """
    latency_ms = int(metadata.pop("latency_ms", 0) or 0)
    return LabEvent(
        node=node,
        event_type=event_type,
        message=message,
        latency_ms=latency_ms,
        metadata=metadata,
    ).model_dump()
