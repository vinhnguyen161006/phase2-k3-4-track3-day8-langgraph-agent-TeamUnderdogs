"""Persistence / crash-recovery demonstration (extension track).

Proves three things the rubric asks for:

1. **thread_id per run** — state is stored under the scenario's own thread.
2. **State history / time travel** — every super-step is replayable.
3. **Cross-process resume** — a *fresh* saver opened on the same SQLite file
   reads back a run written by an earlier graph object, which is what makes
   crash-recovery possible.

Run:  python scripts/demo_persistence.py
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

DB_PATH = "outputs/demo_checkpoints.db"


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path(DB_PATH).unlink(missing_ok=True)

    scenario = Scenario(
        id="persist_demo",
        query="Please lookup order status for order 55512",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}

    # ── Pass 1: run the graph, writing checkpoints to disk ──────────
    saver1 = build_checkpointer("sqlite", DB_PATH)
    graph1 = build_graph(checkpointer=saver1)
    result = graph1.invoke(state, config=config)
    print(f"[run]     thread_id={state['thread_id']}  route={result.get('route')}")
    print(f"[run]     final_answer={str(result.get('final_answer'))[:70]}...")

    history1 = list(graph1.get_state_history(config))
    print(f"[history] {len(history1)} checkpoints written")
    for snap in reversed(history1):
        step = snap.metadata.get("step") if snap.metadata else "?"
        nxt = snap.next or ("END",)
        print(f"          step {step:>3} -> next={nxt}")

    # ── Pass 2: simulate a crash — drop the graph, reopen the DB ────
    del graph1, saver1

    saver2 = build_checkpointer("sqlite", DB_PATH)
    graph2 = build_graph(checkpointer=saver2)
    restored = graph2.get_state(config)

    print("\n[resume]  reopened DB in a fresh saver (simulated restart)")
    print(f"[resume]  recovered route        = {restored.values.get('route')}")
    print(f"[resume]  recovered attempt      = {restored.values.get('attempt')}")
    print(f"[resume]  recovered events       = {len(restored.values.get('events', []))}")
    print(f"[resume]  recovered final_answer = {str(restored.values.get('final_answer'))[:60]}...")

    # ── Time travel: replay from an earlier checkpoint ──────────────
    history2 = list(graph2.get_state_history(config))
    mid = history2[len(history2) // 2]
    print(f"\n[travel]  replaying from step {mid.metadata.get('step')} (next={mid.next})")
    if mid.next:
        replayed = graph2.invoke(None, config=mid.config)
        print(f"[travel]  replay produced route={replayed.get('route')}")
    else:
        print("[travel]  selected checkpoint is terminal; nothing to replay")

    ok = bool(restored.values.get("final_answer")) and len(history2) > 1
    print(f"\nRESUME_SUCCESS = {ok}")


if __name__ == "__main__":
    main()
