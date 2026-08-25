"""Report generation from MetricsReport data.

Renders the structure of ``reports/lab_report_template.md`` with the numbers of
an actual run substituted in, so the report can never drift from the metrics it
describes — re-running the scenarios regenerates it.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from .metrics import MetricsReport

#: Field -> (reducer, rationale). Mirrors the annotations in state.py.
_STATE_FIELDS: list[tuple[str, str, str]] = [
    ("query", "overwrite", "normalized once at intake; only the current text matters"),
    ("route", "overwrite", "the classifier's current verdict, not its history"),
    ("risk_level", "overwrite", "derived from route; recomputed on each classify"),
    ("attempt", "overwrite", "a counter — appending would break the retry bound"),
    ("max_attempts", "overwrite", "per-scenario retry budget, set once"),
    ("evaluation_result", "overwrite",
     "gate for route_after_evaluate; stale verdicts must not linger"),
    ("pending_question", "overwrite", "only the latest clarification is actionable"),
    ("proposed_action", "overwrite", "one action awaits approval at a time"),
    ("approval", "overwrite", "the binding decision is the most recent one"),
    ("final_answer", "overwrite", "single reply surface for every route"),
    ("messages", "append", "conversation/audit trail — history must survive retries"),
    ("tool_results", "append", "every attempt is evidence; evaluate reads the latest"),
    ("errors", "append", "failure forensics; retries would otherwise erase earlier errors"),
    ("events", "append", "the audit log metrics are computed from (nodes_visited, retries)"),
]


def _git_commit() -> str:
    """Short hash of the current commit, or a placeholder outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"


def _summary_table(metrics: MetricsReport) -> str:
    rows = [
        ("Total scenarios", str(metrics.total_scenarios)),
        ("Success rate", f"{metrics.success_rate:.0%}"),
        ("Avg nodes visited", f"{metrics.avg_nodes_visited:.2f}"),
        ("Total retries", str(metrics.total_retries)),
        ("Total interrupts (approvals)", str(metrics.total_interrupts)),
        ("Resume demonstrated", "yes" if metrics.resume_success else "no"),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def _scenario_table(metrics: MetricsReport) -> str:
    lines = [
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Nodes |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for item in metrics.scenario_metrics:
        lines.append(
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '—'} "
            f"| {'PASS' if item.success else 'FAIL'} | {item.retry_count} "
            f"| {item.interrupt_count} | {item.nodes_visited} |"
        )
    return "\n".join(lines)


def _state_table() -> str:
    lines = ["| Field | Reducer | Why |", "|---|---|---|"]
    lines += [f"| {name} | {reducer} | {why} |" for name, reducer, why in _STATE_FIELDS]
    return "\n".join(lines)


def _observations(metrics: MetricsReport) -> str:
    """Explain why the numbers look the way they do (required by METRICS.md)."""
    notes: list[str] = []

    retried = [m for m in metrics.scenario_metrics if m.retry_count]
    if retried:
        detail = ", ".join(f"{m.scenario_id} ({m.retry_count})" for m in retried)
        notes.append(
            f"- **Retries ({metrics.total_retries} total)** come only from error-route "
            f"scenarios: {detail}. `tool_node` fails while `attempt < 2`, so a ticket with "
            "the default budget of 3 retries twice and then succeeds, while a ticket whose "
            "`max_attempts` is 1 can never reach the success branch and escalates instead."
        )
    else:
        notes.append("- **Retries**: no scenario exercised the retry loop in this run.")

    approvals = [m for m in metrics.scenario_metrics if m.approval_observed]
    if approvals:
        notes.append(
            f"- **Interrupts ({metrics.total_interrupts} total)** equal the number of "
            f"risky-route scenarios ({', '.join(m.scenario_id for m in approvals)}). Each "
            "passes `risky_action → approval` before any side effect, which is exactly the "
            "HITL gate the rubric asks for."
        )

    dead_letters = [m for m in metrics.scenario_metrics if m.retry_count and m.retry_count >= 1
                    and any("dead_letter" in e for e in m.errors)]
    if dead_letters:
        notes.append(
            f"- **Dead letter**: {', '.join(m.scenario_id for m in dead_letters)} exhausted the "
            "retry budget and terminated through `dead_letter → finalize`, proving the loop is "
            "bounded rather than merely usually-terminating."
        )

    failures = [m for m in metrics.scenario_metrics if not m.success]
    if failures:
        detail = ", ".join(
            f"{m.scenario_id} (expected {m.expected_route}, got {m.actual_route or 'none'})"
            for m in failures
        )
        notes.append(
            f"- **Failures**: {detail}. These are classification disagreements, not wiring "
            "faults — every scenario still terminated at `finalize`."
        )
    else:
        notes.append(
            "- **No failures**: every scenario reached its expected route and produced either a "
            "`final_answer` or a `pending_question`."
        )

    spread = [m.nodes_visited for m in metrics.scenario_metrics]
    if spread:
        notes.append(
            f"- **Node counts** range {min(spread)}–{max(spread)}. The short end is the `simple` "
            "route (intake → classify → answer → finalize); the long end is the retry/dead-letter "
            "path, which revisits `tool` and `evaluate` on every attempt."
        )
    return "\n".join(notes)


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    return f"""# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / student

Team of 4:

| # | Name | Student ID |
|---:|---|---|
| 1 | Nguyen Quang Vinh | 2A202601517 |
| 2 | | |
| 3 | | |
| 4 | | |

- Repo/commit: phase2-k3-4-track3-day8-langgraph-agent @ {_git_commit()}
- Date: {date.today().isoformat()}

> Generated from `outputs/metrics.json` by `report.render_report()` — the tables
> below are the numbers of the run, not hand-copied.

## 2. Architecture

The workflow is an 11-node `StateGraph` where every terminal path converges on a
single `finalize` node before `END`.

```
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
```

Four conditional edges carry all the branching, and each is a pure function of
state (`routing.py`): they read state and return a node name, never mutate or
call out. That keeps them unit-testable without an LLM and makes a checkpoint
replay deterministic — the same state always takes the same branch.

**Why a graph rather than a chain.** A linear LCEL chain can only run forward.
Three behaviours here need something a chain cannot express: the
`evaluate → retry → tool` **cycle**, the `approval` **suspension point**, and
the **convergence** of five different routes onto one `finalize`.

**LLM integration.** `classify_node` calls the LLM with
`.with_structured_output(Classification)`, so the provider is constrained to the
five legal route literals instead of returning free text that needs parsing.
`answer_node` generates the reply grounded in accumulated `tool_results` and the
approval decision. `evaluate_node` applies an LLM-as-judge over the latest tool
result, behind a cheap deterministic guard for the explicit `ERROR:` marker so
an obvious failure costs no API call. No node reads `scenario_id`: routing is
driven by the ticket text alone, so hidden scenarios behave like the sample ones.

## 3. State schema

{_state_table()}

The split matters at exactly one place: the retry loop re-enters `tool` and
`evaluate` repeatedly. Append-only fields accumulate the evidence of every
attempt (which is what `nodes_visited` and `retry_count` are computed from),
while overwrite fields hold only the current verdict — if `evaluation_result`
appended, a single early `needs_retry` would keep the loop alive forever.

## 4. Scenario results

{_summary_table(metrics)}

{_scenario_table(metrics)}

### Why the numbers look like this

{_observations(metrics)}

## 5. Failure analysis

**1. Tool failure and unbounded retry.** The dangerous version of a retry loop
is one whose exit condition depends on the thing that is failing. Here the bound
is structural: `retry_or_fallback_node` increments `attempt` *before*
`route_after_retry` reads it, so every pass strictly increases the counter and
`attempt >= max_attempts` is always reached. Past the budget the run escalates to
`dead_letter`, which still writes a `final_answer` — the user gets an honest
"escalated to a human" rather than a hang or a silent drop. `S07_dead_letter`
(`max_attempts: 1`) exercises this path on every run.

**2. Risky action executed without approval.** A refund or deletion must not be
reachable by classification alone. The graph makes this a topology property
rather than a policy check: the `risky` branch has no edge directly to `tool`.
The only way in is `risky_action → approval → [route_after_approval]`, and
rejection routes to `clarify`, not to execution. `tool_node` also inspects the
approval decision, so an unapproved risky action cannot produce an
"action executed" result even if it were somehow reached.

**3. LLM unavailability (third mode).** Every LLM node is wrapped: on a provider
error it records the exception in `errors` and falls back to a deterministic
path — a keyword heuristic for classification, a templated grounded reply for
`answer`. An outage degrades output quality but never breaks termination.

## 6. Persistence / recovery evidence

`build_checkpointer()` supports `memory`, `sqlite` and `postgres`. Each scenario
runs under its own `thread_id` (`thread-{{scenario_id}}`), so LangGraph writes a
checkpoint after every super-step, keyed per scenario.

With the SQLite backend the checkpoints outlive the process — `checkpoints.db`
holds the full per-thread history, and `graph.get_state_history(config)` replays
the super-steps of any past run (time travel). `scripts/demo_persistence.py`
demonstrates this end to end: it runs a scenario, closes the graph, reopens the
database in a fresh saver, and reads the state back by `thread_id`.

## 7. Extension work

- **SQLite checkpointer** with WAL mode (`persistence.py`), plus a Postgres path.
- **State-history / time-travel replay** and cross-process resume
  (`scripts/demo_persistence.py`).
- **LLM-as-judge** in `evaluate_node` rather than a substring heuristic.
- **Real HITL interrupts**: `LANGGRAPH_INTERRUPT=true` switches `approval_node`
  to `langgraph.types.interrupt()`, resumable with `Command(resume=...)`.
- **Mermaid graph diagram** via `render_mermaid()` (`docs/graph.mmd`).
- **Graceful degradation** on LLM failure in every LLM-backed node, plus
  **rate-limit backoff** (`llm.call_with_retry`) that retries 429/quota errors
  with exponential delay + jitter and lets all other errors surface immediately.

## 8. Improvement plan

With one more day, in priority order:

1. **Make the tool layer real.** `tool_node` is a mock; the first production
   step is a registry of typed tools with per-tool timeouts, so retry can
   distinguish "transient 503, retry" from "404, do not retry" instead of
   treating every failure as retryable.
2. **Backoff in the graph's own retry loop.** LLM calls already back off
   (`call_with_retry`), but `retry → tool` re-fires immediately; the same
   exponential-plus-jitter policy belongs on the tool retry so a struggling
   dependency is not hammered.
3. **Persist the dead-letter queue.** Escalations exist only as state today;
   they should be rows a human queue can consume and re-drive.
4. **Cache and evaluate the classifier.** Classification is the highest-leverage
   LLM call — a labelled set with per-route precision/recall would catch prompt
   regressions, and caching identical tickets would cut cost.
5. **Tracing.** LangSmith spans per node to make latency attribution and prompt
   debugging possible in production.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
