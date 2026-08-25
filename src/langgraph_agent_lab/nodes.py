"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM usage in this implementation:
- ``classify_node``  — real LLM call, structured output (Pydantic) for intent routing.
- ``answer_node``    — real LLM call, grounded generation over tool_results/approval.
- ``evaluate_node``  — real LLM call, LLM-as-judge over the latest tool result.

None of these nodes look at ``scenario_id``: routing is driven purely by the
LLM's reading of ``query`` plus state logic, so unseen scenarios behave the same
way the sample ones do.

Every LLM node degrades gracefully. If the provider errors out (no key, rate
limit, transient 5xx) the node records the failure in ``errors`` and falls back
to a deterministic path, so the graph always terminates instead of raising
mid-run and losing the whole scenario.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import call_with_retry, get_llm
from .state import CLASSIFIABLE_ROUTES, AgentState, ApprovalDecision, Route, make_event


# ─── LLM structured-output schemas ───────────────────────────────────
class Classification(BaseModel):
    """Structured classification returned by the LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best route for this support ticket."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="How damaging an incorrect automated action would be.",
    )
    reason: str = Field(default="", description="One short sentence justifying the route.")


class ToolJudgement(BaseModel):
    """LLM-as-judge verdict on a tool result."""

    satisfactory: bool = Field(
        description="True if the tool result answers the request and contains no failure."
    )
    reason: str = Field(default="", description="One short sentence explaining the verdict.")


CLASSIFY_SYSTEM_PROMPT = """You are the intent router for a customer-support agent.
Classify the ticket into EXACTLY ONE route.

Routes, in strict priority order — pick the FIRST one that applies:

1. risky — the user asks you to PERFORM an action with real side effects:
   refunds, deletions, cancellations, sending email, charging or moving money,
   resetting or changing someone else's account, closing a subscription.
   These need human approval before execution.
2. tool  — the user asks you to LOOK UP or retrieve information: order status,
   tracking numbers, account details, search, "where is my X", "check Y".
   Read-only. No side effects.
3. missing_info — the request is too vague to act on: no object, no identifier,
   no context. Examples: "can you fix it?", "it's broken", "help me please".
   You would have to guess what the user means.
4. error — the ticket REPORTS a system/technical failure: timeout, crash,
   service unavailable, "cannot recover", "system failure", exception traces.
5. simple — a general question answerable from knowledge alone, with no lookup
   and no action. Examples: "how do I reset my password?", "what are your hours?"

Decision aids:
- Asking to DO something destructive -> risky (even if phrased politely).
- Asking to KNOW something -> tool.
- A failure being reported to you -> error. A failure you must fix for a
  specific named resource -> still error unless a side-effecting action is named.
- If the ticket names no specific object AND no specific action -> missing_info.

Set risk_level to "high" for the risky route, "low" otherwise.
Answer with the structured schema only."""

ANSWER_SYSTEM_PROMPT = """You are a customer-support agent writing the final reply.

Ground your answer ONLY in the context provided (tool results, approval
decision, original ticket). Rules:
- If tool results are present, base the answer on them and cite the concrete
  values they contain. Do not invent order numbers, dates, or amounts.
- If an approval decision is present and approved, confirm the action was
  authorised and carried out.
- If the context is thin, answer the question generally and say what you would
  need to be more specific. Never fabricate account-specific facts.
- 2-4 sentences. Plain, direct, no greeting boilerplate, no sign-off."""

CLARIFY_SYSTEM_PROMPT = """You are a customer-support agent. The user's request is
too vague to act on. Ask ONE specific clarifying question that would unblock you.

Name the concrete detail you need (which order, which account, which product,
what exactly went wrong). One sentence, ending in a question mark. Do not
apologise, do not guess what they meant, do not offer a solution yet."""


def _latest(items: list[str] | None) -> str:
    """Return the last entry of an append-only list, or '' when empty."""
    return items[-1] if items else ""


def _text_of(response: Any) -> str:
    """Extract plain text from a LangChain chat response."""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content blocks
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content).strip()


def _heuristic_route(query: str) -> tuple[str, str]:
    """Deterministic fallback used only when the LLM call fails.

    This is NOT the primary classifier — ``classify_node`` always tries the LLM
    first. It exists so an API outage degrades to a sensible route and a
    terminating graph rather than an exception. Same priority order as the
    prompt: risky > tool > missing_info > error > simple.
    """
    text = query.lower()
    risky = ("refund", "delete", "cancel", "remove", "charge", "send confirmation", "close account")
    lookup = ("lookup", "look up", "status", "track", "find", "search", "check", "where is")
    failure = ("timeout", "crash", "failure", "unavailable", "cannot recover", "exception")

    if any(word in text for word in risky):
        return Route.RISKY.value, "high"
    if any(word in text for word in lookup):
        return Route.TOOL.value, "low"
    if any(word in text for word in failure):
        return Route.ERROR.value, "low"
    if len(text.split()) <= 4:
        return Route.MISSING_INFO.value, "low"
    return Route.SIMPLE.value, "low"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── LLM-backed nodes ────────────────────────────────────────────────
def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output.

    Uses ``.with_structured_output(Classification)`` so the provider is
    constrained to the five valid route literals — no brittle text parsing, and
    unseen phrasings still land on a legal route.
    """
    query = state.get("query", "")
    started = time.perf_counter()

    # Widened to str: the LLM branch yields Literal values, the fallback a plain
    # str, and both must flow into the same state field.
    route: str
    risk_level: str

    try:
        llm = get_llm().with_structured_output(Classification)
        result: Classification = call_with_retry(
            lambda: llm.invoke(
                [
                    ("system", CLASSIFY_SYSTEM_PROMPT),
                    ("human", f"Support ticket:\n{query}"),
                ]
            )
        )
        route = result.route
        risk_level = "high" if route == Route.RISKY.value else result.risk_level
        reason = result.reason
        source = "llm"
        errors: list[str] = []
    except Exception as exc:  # provider outage / missing key / bad response
        route, risk_level = _heuristic_route(query)
        reason = f"LLM classification unavailable ({type(exc).__name__}); used fallback."
        source = "fallback"
        errors = [f"classify: {type(exc).__name__}: {exc}"]

    if route not in CLASSIFIABLE_ROUTES:  # defensive: never emit an unknown route
        route, risk_level = _heuristic_route(query)
        source = "fallback"

    latency_ms = int((time.perf_counter() - started) * 1000)
    update: dict[str, Any] = {
        "route": route,
        "risk_level": risk_level,
        "classify_reason": reason,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"routed to {route}",
                route=route,
                risk_level=risk_level,
                source=source,
                reason=reason,
                latency_ms=latency_ms,
            )
        ],
    }
    if errors:
        update["errors"] = errors
    return update


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, simulating transient failures on the error route.

    The failure is a function of ``attempt``, not of the scenario id: an
    error-route ticket fails while ``attempt < 2`` and succeeds afterwards. That
    exercises the retry loop for normal scenarios, while a scenario whose
    ``max_attempts`` is below 2 can never reach the success branch and correctly
    falls through to the dead-letter node.
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt} (service unavailable)"
        event_type = "failed"
    else:
        approval = state.get("approval") or {}
        if approval.get("approved"):
            result = (
                f"OK: approved action executed for '{query[:60]}' "
                f"(reference TCK-{abs(hash(query)) % 10000:04d}, status=completed)"
            )
        else:
            result = (
                f"OK: lookup completed for '{query[:60]}' "
                f"(reference TCK-{abs(hash(query)) % 10000:04d}, status=in_transit, "
                f"last_update=2 hours ago)"
            )
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:attempt={attempt}"],
        "events": [
            make_event("tool", event_type, result[:80], attempt=attempt, route=route)
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate (LLM-as-judge).

    Asks the LLM whether the result actually satisfies the request. A cheap
    deterministic guard runs first for the unambiguous failure marker, so an
    obvious ``ERROR:`` never costs an API call and never depends on the judge
    being available.
    """
    latest = _latest(state.get("tool_results"))

    if not latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "failed", "no tool result to evaluate")],
        }

    # Fast path: explicit failure marker needs no judge.
    if latest.upper().startswith("ERROR"):
        return {
            "evaluation_result": "needs_retry",
            "messages": ["evaluate:needs_retry"],
            "events": [
                make_event(
                    "evaluate", "completed", "tool reported an error", source="guard",
                    verdict="needs_retry",
                )
            ],
        }

    try:
        judge = get_llm().with_structured_output(ToolJudgement)
        verdict: ToolJudgement = call_with_retry(
            lambda: judge.invoke(
                [
                    (
                        "system",
                        "You are a strict QA judge for a support agent's tool calls. "
                        "Decide whether the tool result is usable as the basis for a "
                        "reply to the user. Mark it unsatisfactory if it reports an "
                        "error, is empty, or does not address the request.",
                    ),
                    (
                        "human",
                        f"User request:\n{state.get('query', '')}\n\nTool result:\n{latest}",
                    ),
                ]
            )
        )
        evaluation = "success" if verdict.satisfactory else "needs_retry"
        return {
            "evaluation_result": evaluation,
            "messages": [f"evaluate:{evaluation}"],
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    verdict.reason or evaluation,
                    source="llm-judge",
                    verdict=evaluation,
                )
            ],
        }
    except Exception as exc:
        # Judge unavailable: the result carried no error marker, so accept it
        # rather than spinning the retry loop on an infrastructure problem.
        return {
            "evaluation_result": "success",
            "errors": [f"evaluate: {type(exc).__name__}: {exc}"],
            "events": [
                make_event(
                    "evaluate", "completed", "judge unavailable; accepted result",
                    source="fallback", verdict="success",
                )
            ],
        }


def answer_node(state: AgentState) -> dict:
    """Generate the final response with an LLM, grounded in accumulated context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results") or []
    approval = state.get("approval") or {}
    started = time.perf_counter()

    context_parts = [f"Original ticket: {query}"]
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(f"- {r}" for r in tool_results))
    if approval:
        context_parts.append(
            f"Approval decision: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} — {approval.get('comment')}"
        )
    if state.get("proposed_action"):
        context_parts.append(f"Action performed: {state['proposed_action']}")
    context = "\n\n".join(context_parts)

    try:
        llm = get_llm(temperature=0.2)
        answer = _text_of(
            call_with_retry(
                lambda: llm.invoke([("system", ANSWER_SYSTEM_PROMPT), ("human", context)])
            )
        )
        source = "llm"
        errors: list[str] = []
    except Exception as exc:
        summary = _latest(tool_results) or "no tool output was required"
        answer = (
            f"Regarding '{query[:80]}': {summary}. "
            "A support specialist will follow up if you need more detail."
        )
        source = "fallback"
        errors = [f"answer: {type(exc).__name__}: {exc}"]

    if not answer:  # provider returned empty content
        answer = f"Regarding '{query[:80]}': your request has been processed."
        source = "fallback"

    latency_ms = int((time.perf_counter() - started) * 1000)
    update: dict[str, Any] = {
        "final_answer": answer,
        "messages": [f"answer:{answer[:40]}"],
        "events": [
            make_event(
                "answer", "completed", answer[:80],
                source=source, grounded_in=len(tool_results), latency_ms=latency_ms,
            )
        ],
    }
    if errors:
        update["errors"] = errors
    return update


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating an answer.

    Also serves the rejected-approval path: when a human declines a risky
    action, the user is asked how they would like to proceed instead.
    """
    query = state.get("query", "")
    approval = state.get("approval") or {}
    rejected = approval and not approval.get("approved")

    try:
        if rejected:
            human = (
                f"The request '{query}' was declined by a reviewer "
                f"({approval.get('comment') or 'no reason given'}). Ask the user "
                "one question about how they would like to proceed instead."
            )
        else:
            human = f"Vague support ticket:\n{query}"
        llm = get_llm(temperature=0.2)
        question = _text_of(
            call_with_retry(
                lambda: llm.invoke([("system", CLARIFY_SYSTEM_PROMPT), ("human", human)])
            )
        )
        source = "llm"
        errors: list[str] = []
    except Exception as exc:
        question = (
            "Could you tell me which account or order this refers to, and what "
            "exactly you would like changed?"
        )
        source = "fallback"
        errors = [f"clarify: {type(exc).__name__}: {exc}"]

    if not question:
        question = "Could you share a few more details so I can help with this?"
        source = "fallback"

    update: dict[str, Any] = {
        "pending_question": question,
        # pending_question alone satisfies the metrics' output check, but a
        # final_answer keeps the reply surface uniform across every route.
        "final_answer": question,
        "messages": [f"clarify:{question[:40]}"],
        "events": [
            make_event(
                "clarify", "completed", question[:80],
                source=source, after_rejection=bool(rejected),
            )
        ],
    }
    if errors:
        update["errors"] = errors
    return update


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Records *what* would happen and *why* it needs sign-off, so the approver (or
    the audit log) sees the side effect before it is executed.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    reason = state.get("classify_reason") or "the request performs an irreversible side effect"

    proposed = (
        f"Proposed action: {query.strip()} | risk={risk_level} | "
        f"requires human approval because {reason}"
    )
    return {
        "proposed_action": proposed,
        "messages": [f"risky_action:{query[:40]}"],
        "events": [
            make_event(
                "risky_action", "pending_approval", proposed[:80],
                risk_level=risk_level,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: a mock reviewer approves, so tests and CI run offline and
    deterministically. Set ``LANGGRAPH_INTERRUPT=true`` to suspend the graph
    with ``interrupt()`` and wait for a real decision resumed via
    ``Command(resume=...)``.
    """
    proposed = state.get("proposed_action") or state.get("query", "")
    interrupt_enabled = os.getenv("LANGGRAPH_INTERRUPT", "").lower() in {"1", "true", "yes"}
    mode = "mock"

    decision = ApprovalDecision(
        approved=True,
        reviewer="mock-reviewer",
        comment="Auto-approved by mock reviewer (offline default).",
    )

    if interrupt_enabled:
        try:
            from langgraph.types import interrupt

            raw = interrupt(
                {
                    "type": "approval_request",
                    "proposed_action": proposed,
                    "risk_level": state.get("risk_level", "high"),
                    "question": "Approve this action? Reply with {'approved': true|false}",
                }
            )
            mode = "interrupt"
            if isinstance(raw, dict):
                decision = ApprovalDecision(
                    approved=bool(raw.get("approved", False)),
                    reviewer=str(raw.get("reviewer", "human-reviewer")),
                    comment=str(raw.get("comment", "")),
                )
            else:
                decision = ApprovalDecision(
                    approved=bool(raw), reviewer="human-reviewer", comment=str(raw)
                )
        except ImportError:
            mode = "mock"

    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{decision.approved}"],
        "events": [
            make_event(
                "approval",
                "approved" if decision.approved else "rejected",
                f"{decision.reviewer}: {decision.comment}"[:80],
                approved=decision.approved,
                mode=mode,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt and increment the bounded attempt counter."""
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    latest = _latest(state.get("tool_results")) or state.get("query", "")

    message = f"retry {attempt}/{max_attempts} after: {latest[:60]}"
    return {
        "attempt": attempt,
        "errors": [message],
        "messages": [f"retry:{attempt}"],
        "events": [
            make_event(
                "retry", "retrying", message[:80],
                attempt=attempt, max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after the retry budget is exhausted.

    Third layer of the retry -> fallback -> dead-letter ladder: nothing is
    silently dropped, the user gets an honest answer and the event trail records
    the escalation.
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    query = state.get("query", "")

    answer = (
        f"I could not complete this request after {attempt} attempt(s) "
        f"(limit {max_attempts}). The ticket '{query[:60]}' has been escalated to "
        "a human support engineer, who will contact you directly."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "errors": [f"dead_letter: exhausted {attempt}/{max_attempts} attempts"],
        "events": [
            make_event(
                "dead_letter", "escalated", answer[:80],
                attempt=attempt, max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit the final audit event. All routes pass through here before END."""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                attempts=int(state.get("attempt", 0)),
                answered=bool(state.get("final_answer") or state.get("pending_question")),
                approval_observed=state.get("approval") is not None,
            )
        ]
    }
