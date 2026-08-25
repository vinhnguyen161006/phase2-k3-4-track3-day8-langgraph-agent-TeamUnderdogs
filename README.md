# Day 08 Lab — LangGraph Agentic Orchestration

Build a production-style LangGraph workflow for a support-ticket agent with state management, conditional routing, retry loops, human-in-the-loop approval, persistence, and metrics.

This is a **starter skeleton**. All node implementations, routing logic, and graph wiring are left as `TODO(student)` — you must build them from scratch.

---

## How you will be graded

| Category | Points | What we look for |
|---|---:|---|
| Architecture & state schema | 15 | Typed state with correct reducers, student-added fields, lean serializable state |
| Graph construction & wiring | 15 | All nodes registered, edges correct, conditional edges work, graph compiles |
| LLM integration | 15 | classify_node + answer_node use real LLM calls (structured output, grounded generation) |
| Graph behavior | 20 | All scenario routes correct, bounded retry loop, HITL approval path, all routes terminate |
| Persistence & recovery | 10 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics & tests | 15 | `metrics.json` valid, scenario coverage, tests pass, meaningful counts |
| Report & demo | 10 | Architecture explanation, metrics table, failure analysis, improvement ideas |

**Grade bands:**
- **90–100**: Production-quality graph + LLM integration + metrics + report + at least one bonus extension
- **75–89**: Core graph works with LLM, metrics valid, report explains trade-offs
- **60–74**: Graph mostly works but LLM integration, persistence, or report incomplete
- **< 60**: Does not run, hard-codes scenarios, or lacks LLM integration/metrics/report

> **Critical rule**: Do NOT hard-code answers to specific scenario queries. Your graph must route based on **LLM classification and state logic**, not by matching exact scenario IDs. We grade with additional hidden scenarios.

---

## LLM Integration Requirements

This lab requires real LLM API calls in specific nodes:

| Node | Requirement | Pattern |
|---|---|---|
| `classify_node` | **MUST use LLM** | Structured output (`.with_structured_output()`) for intent classification |
| `answer_node` | **MUST use LLM** | Grounded response generation using tool_results/context |
| `evaluate_node` | **SHOULD use LLM** (bonus) | LLM-as-judge to evaluate tool results quality |

A helper is provided in `src/langgraph_agent_lab/llm.py` — it reads your API key from `.env` and returns a LangChain chat model.

```bash
# Install your preferred LLM provider
pip install langchain-openai    # for OpenAI
# OR
pip install langchain-anthropic  # for Anthropic

# Configure .env
cp .env.example .env
# Edit .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## Understanding `scenarios.jsonl`

The file `data/sample/scenarios.jsonl` contains **7 sample scenarios** your graph must handle:

```jsonl
{"id":"S01_simple",      "query":"How do I reset my password?",                          "expected_route":"simple"}
{"id":"S02_tool",        "query":"Please lookup order status for order 12345",            "expected_route":"tool"}
{"id":"S03_missing",     "query":"Can you fix it?",                                      "expected_route":"missing_info"}
{"id":"S04_risky",       "query":"Refund this customer and send confirmation email",      "expected_route":"risky"}
{"id":"S05_error",       "query":"Timeout failure while processing request",              "expected_route":"error"}
{"id":"S06_delete",      "query":"Delete customer account after support verification",    "expected_route":"risky"}
{"id":"S07_dead_letter", "query":"System failure cannot recover after multiple attempts", "expected_route":"error", "max_attempts":1}
```

### What each field means

| Field | Purpose |
|---|---|
| `id` | Unique scenario identifier — used in metrics output |
| `query` | The user's support-ticket text — input to your graph |
| `expected_route` | Which route your `classify_node` should pick: `simple`, `tool`, `missing_info`, `risky`, or `error` |
| `requires_approval` | If `true`, your graph must hit the approval/HITL node before answering |
| `should_retry` | If `true`, scenario simulates transient tool failure requiring retry |
| `max_attempts` | Override retry limit (default 3). S07 sets this to 1, so it exhausts retries immediately → dead letter |
| `tags` | Descriptive labels for your reference |

### How scenarios flow through your code

```
scenarios.jsonl  →  scenarios.py loads them  →  cli.py runs each through your graph
                                              →  metrics.py collects results
                                              →  outputs/metrics.json
```

1. `make run-scenarios` reads `data/sample/scenarios.jsonl`
2. For each scenario, it calls `initial_state(scenario)` → `graph.invoke(state)`
3. After execution, it checks: did `actual_route` match `expected_route`? Did HITL fire when required?
4. Results go to `outputs/metrics.json`

### How to design your classification

Your `classify_node` should use an LLM to classify intent. Design a prompt that routes queries:

| Route | Intent |
|---|---|
| `risky` | Actions with side effects: refunds, deletions, sending emails, cancellations |
| `tool` | Information lookups: order status, tracking, search queries |
| `missing_info` | Vague/incomplete queries lacking actionable context |
| `error` | System failures: timeouts, crashes, service unavailable |
| `simple` | General questions answerable without tools or actions |

**Priority matters**: risky > tool > missing_info > error > simple. Design your LLM prompt to respect this priority.

### Adding your own test scenarios

You can add extra lines to `scenarios.jsonl` to test edge cases:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

The grading script will also test with scenarios you haven't seen.

---

## Quick start

```bash
# Option A: conda
conda activate ai-lab
pip install -e '.[dev]'
pip install langchain-openai  # or langchain-anthropic

# Option B: venv
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install langchain-openai  # or langchain-anthropic

# Configure LLM
cp .env.example .env
# Edit .env — set your API key

# Verify setup
make test  # some tests will fail until you implement TODOs
```

---

## Step-by-step workflow

### Phase 1: State + nodes (0–90 min) — worth 30 points

1. **`state.py`** — Review existing fields. Add missing fields as you discover them:
   - `evaluation_result` for retry loop gate
   - `pending_question` for clarification flow
   - `proposed_action` for risky action flow
   - `approval` for HITL decisions

2. **`llm.py`** — Review the helper. Configure `.env` with your API key.

3. **`nodes.py`** — Implement all 10 node functions:
   - `classify_node`: **LLM + structured output** for intent classification
   - `tool_node`: mock tool with error simulation
   - `evaluate_node`: tool result quality check (LLM-as-judge for bonus)
   - `answer_node`: **LLM-generated** grounded response
   - `ask_clarification_node`: generate clarification question
   - `risky_action_node`: prepare action for approval
   - `approval_node`: mock approval with optional interrupt()
   - `retry_or_fallback_node`: increment attempt counter
   - `dead_letter_node`: handle max retry exhaustion
   - `finalize_node`: emit final audit event

### Phase 2: Routing + graph (90–150 min) — worth 35 points

4. **`routing.py`** — Implement all 4 routing functions from scratch
5. **`graph.py`** — Build the complete StateGraph:
   - Import and register all 11 nodes
   - Wire fixed + conditional edges
   - All paths must terminate at finalize → END
6. **Verify**: `make test` and `make run-scenarios`

### Phase 3: Persistence (150–180 min) — worth 10 points

7. **`persistence.py`** — Implement SQLite checkpointer
   - Show evidence: thread_id per run, state history, or crash-resume

### Phase 4: Metrics & report (180–240 min) — worth 25 points

8. **`report.py`** — Implement `render_report()` from metrics data
9. **Run**: `make run-scenarios` → `outputs/metrics.json`
10. **Validate**: `make grade-local`
11. **Report**: Fill `reports/lab_report.md`

### Phase 5: Extensions (240+ min) — push toward 90+

Pick one or more:
- **Parallel fan-out**: Use `Send()` for concurrent tool calls
- **Real HITL**: `LANGGRAPH_INTERRUPT=true` with `interrupt()`
- **Streamlit UI**: Build approval/reject interface
- **Time travel**: `get_state_history()` replay
- **Crash recovery**: SQLite checkpoint survives process kill
- **Graph diagram**: `graph.get_graph().draw_mermaid()`

---

## Make commands

| Command | What it does |
|---|---|
| `make install` | Install project + dev dependencies |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy type checker |
| `make run-scenarios` | Execute all scenarios → `outputs/metrics.json` |
| `make grade-local` | Validate metrics.json schema |
| `make clean` | Remove caches and generated files |

---

## Submission checklist

- [x] All `TODO(student)` sections implemented
- [x] `.env` configured with LLM API key (`GEMINI_API_KEY`, `LLM_MODEL=gemini-3.5-flash`)
- [x] `classify_node` uses real LLM call with structured output
- [x] `answer_node` uses real LLM call for grounded responses
- [x] `make test` passes (`make lint` and `make typecheck` clean too)
- [x] `make run-scenarios` generates valid `outputs/metrics.json` (7/7, 100%)
- [x] `make grade-local` passes validation
- [x] `reports/lab_report.md` generated from live metrics
- [x] Can explain at least one route and one failure mode during demo

**For 90+ points, also include:**
- [x] Bonus extensions: SQLite checkpointer, time-travel replay + cross-process
      resume, LLM-as-judge `evaluate_node`, real `interrupt()` HITL mode,
      Mermaid diagram, rate-limit backoff
- [x] Evidence in report §6/§7, `docs/graph.mmd`, `scripts/demo_persistence.py`

### Notes for the grader

- **Model**: `gemini-2.5-flash` (the `.env.example` default) is no longer served
  to new API keys and returns 404, so `.env` pins `LLM_MODEL=gemini-3.5-flash`.
  Any chat model with structured-output support works.
- **Live-LLM tests**: `tests/test_graph_smoke.py` makes real API calls and takes
  several minutes. `tests/test_nodes.py` and `tests/test_graph_behavior.py`
  cover the same logic offline in seconds.
- **Persistence demo**: `python scripts/demo_persistence.py`.

---

## Common pitfalls

1. **Missing state fields**: The starter intentionally omits some fields from `AgentState`. You must add `evaluation_result`, `pending_question`, `proposed_action`, and `approval` as you implement nodes that need them.

2. **LLM structured output**: Use `.with_structured_output(YourModel)` to get reliable classification. Raw text parsing is fragile and will fail on hidden test scenarios.

3. **Unbounded retry**: Always check `attempt < max_attempts` in `route_after_retry`. Without this bound, error scenarios loop forever.

4. **Graph wiring**: Every path must end at `finalize → END`. Missing this means the graph hangs for some scenarios.

5. **SqliteSaver API**: In `langgraph-checkpoint-sqlite` 3.x, use `SqliteSaver(conn=sqlite3.connect(...))` not `SqliteSaver.from_conn_string()`.

6. **API key not set**: If you get "No LLM API key found", check your `.env` file and make sure it's loaded (use `python-dotenv` or export manually).
