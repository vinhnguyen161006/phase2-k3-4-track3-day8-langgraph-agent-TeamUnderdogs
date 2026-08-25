"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph, render_mermaid
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _load_env() -> None:
    """Load .env so API keys are available without exporting them manually."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv()


def _verify_resume(graph: Any, thread_id: str) -> bool:
    """Confirm the checkpointer persisted a replayable history for a thread.

    Evidence for the persistence track: reads the saved state back by
    ``thread_id`` after the run finished, and checks the super-step history is
    walkable (the basis of time-travel replay and crash-resume).
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        if not snapshot or not snapshot.values:
            return False
        history = list(graph.get_state_history(config))
        return len(history) > 1 and bool(snapshot.values.get("events"))
    except Exception:
        return False


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    _load_env()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    last_thread_id = ""
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        last_thread_id = state["thread_id"]
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval
            )
        )
        typer.echo(
            f"  {scenario.id}: expected={scenario.expected_route.value} "
            f"actual={final_state.get('route')}"
        )
    resume_success = bool(checkpointer) and _verify_resume(graph, last_thread_id)
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(
        f"Wrote metrics to {output} "
        f"(success_rate={report.success_rate:.0%}, resume_success={resume_success})"
    )


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("draw-graph")
def draw_graph(
    output: Annotated[Path, typer.Option("--output")] = Path("docs/graph.mmd"),
) -> None:
    """Write the compiled graph as a Mermaid diagram."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_mermaid(), encoding="utf-8")
    typer.echo(f"Wrote graph diagram to {output}")


if __name__ == "__main__":
    app()
