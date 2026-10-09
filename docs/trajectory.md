# Agent Trajectory Observability

GenAI Toolkit captures the full **trajectory** of an agent run — agent → LLM
calls → tool calls → skill loads → human-in-the-loop marks — as a first-class,
local, agent-readable record, and exposes it through the `cli trajectory`
command group and the evaluation stack.

This is built on [NVIDIA NeMo Relay](https://docs.nvidia.com/nemo/relay),
which emits the canonical **Agent Trajectory Observability Format (ATOF)**
event stream from the instrumented agent flow. The local trajectory store is
the **source of truth**; remote backends (LangFuse / OTel / LangSmith) fan out
as projections of the same stream.

## How it works

```
agent run
   │  (factory injects NemoRelayDeepAgentsMiddleware + callback handler)
   ▼
NeMo Relay runtime ── ATOF event stream ──┬── local trajectory store (source of truth)
                                          │     data/trajectories/<run_id>/
                                          │       events.jsonl   (raw ATOF)
                                          │       meta.json      (run summary)
                                          │     index.jsonl       (one line per run)
                                          │
                                          ├── cli trajectory ... (list/show/replay/export/diff/...)
                                          ├── evals              (judge the captured trajectory)
                                          └── remote projections (LangFuse / OTel — Phase 1)
```

Every Deep Agents run (`type: deep` profile) is automatically instrumented:

1. The factory (`_create_deep_agent`) wraps the `create_deep_agent(...)` kwargs
   through `add_nemo_relay_integration(...)`, which appends
   `NemoRelayDeepAgentsMiddleware` — routing model and tool calls through
   Relay `llm` / `tool` scopes.
2. The harness / agent invoke config attaches
   `NemoRelayDeepAgentsCallbackHandler`, mapping the LangGraph run hierarchy
   to Relay agent scopes and emitting human-in-the-loop interrupt/resume marks.
3. A manual ATOF subscriber writes each event to the per-session store and
   aggregates a run summary (`meta.json` + `index.jsonl`).

No code change is needed in agent profiles — instrumentation is automatic
when `nemo-relay[deepagents]` is installed (it degrades to a no-op otherwise).

## The trajectory store

Location: `<data_root>/trajectories/` (from `paths.data_root` in config).

```
data/trajectories/
  <run_id>/                   # run_id = root agent scope UUID
    events.jsonl               # raw ATOF 0.1 event stream (one JSON object per line)
    meta.json                  # run summary (profile, model, tokens, tools, skills, status)
  index.jsonl                  # append-only: one line per run
```

`meta.json` fields: `run_id`, `profile`, `started_at`, `ended_at`, `status`,
`n_llm_calls`, `n_tool_calls`, `total_prompt_tokens`,
`total_completion_tokens`, `tools`, `skills_loaded`, `events_path`.

The read layer is `genai_tk.extra.monitoring.trajectory_store.TrajectoryStore`, which
parses ATOF events into typed `Trajectory` / `LlmCall` / `ToolCall` /
`SkillLoad` objects and projects a run to OpenAI-format messages.

## CLI: `cli trajectory`

| Command | Purpose |
|---|---|
| `cli trajectory list [--profile P] [--since WHEN] [--status failed]` | List recorded runs (id, profile, model, started, LLM/tool counts, tokens, status). |
| `cli trajectory show <id> [--format tree\|json\|messages\|dot\|tui] [--tui]` | Render a trajectory. `tree` = intertwined LLM & tool turn tree; `messages` = OpenAI-format; `dot` = scope graph; `tui` = interactive explorer. |
| `cli trajectory tui [id]` | Interactive full-screen Textual TUI navigator for browsing trajectories, turns, LLM thoughts, and tool logs. |
| `cli trajectory tail [--n 20]` | Last N ATOF events from the most recent run. |
| `cli trajectory replay <id> [--delay 0.5]` | Replay events in order with relative timings. |
| `cli trajectory export <id> --format atif\|atof\|messages\|otel\|langfuse [--out file]` | Export a trajectory (ATOF JSONL, ATIF RFC-0001, OpenAI messages, OTEL spans, or Langfuse trace descriptor). |
| `cli trajectory open <id> [--backend langfuse\|harbor]` | Open the recorded run directly in the Langfuse UI or Harbor viewer. |
| `cli trajectory link <id>` | Display the direct Langfuse URL and Harbor command for a recorded run. |
| `cli trajectory diff <id1> <id2>` | Structural diff (tools, skills, step counts, tokens). |
| `cli trajectory skills <id>` | Show `skill.load` marks and where they occurred. |
| `cli trajectory stats [--since WHEN]` | Aggregate: token totals, tool/skill frequency, failure rate, latency p50/p95. |
| `cli trajectory prune [--keep-last N] [--older-than DAYS]` | Retention. |
| `cli trajectory view` | Launch the [Harbor](https://www.harborframework.com/) ATIF web viewer on the store (no-op if `harbor` isn't installed). |

```bash
# List recent runs
uv run cli trajectory list

# Inspect one run as a scope timeline
uv run cli trajectory show <run_id>

# Inspect and simultaneously open in Langfuse
uv run cli trajectory show <run_id> --open

# Open directly in Langfuse
uv run cli trajectory open <run_id> --backend langfuse

# Get trace links
uv run cli trajectory link <run_id>

# Export the captured trajectory as OpenAI messages for offline eval
uv run cli trajectory export <run_id> --format messages --out run.json

# See which skills were loaded
uv run cli trajectory skills <run_id>

# Launch the Harbor web viewer (uv tool install harbor)
uv run cli trajectory view
```

## Programmatic access

```python
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore

store = TrajectoryStore()

# List runs
for run in store.list_runs(profile="docgraph"):
    print(run.run_id, run.profile, run.n_tool_calls, run.tools)

# Reconstruct one run
traj = store.get("<run_id>")
print(traj.llm_calls[0].model, traj.llm_calls[0].usage)
print(traj.tool_names, traj.skill_names)

# Project to OpenAI messages (for agentevals/openevals)
messages = store.messages("<run_id>")
```

## Deterministic Trajectory Matching & Decision Model Evaluations

Store-based evaluation loads a **captured** trajectory and evaluates it — no agent
re-run.

### 1. Deterministic Tool Sequence Matching (Zero LLM, 100% Deterministic)

Instead of depending on external packages like `agentevals`, the toolkit provides native
tool matching over recorded trajectories:

```python
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore, match_trajectory_tools

store = TrajectoryStore()
traj = store.get("<run_id>")

# Using Trajectory helper
assert traj.match_tools(["python_interpreter"], mode="superset")
assert traj.match_tools(["python_interpreter"], mode="strict")

# Standalone function
tools_called = [tc.name for tc in traj.tool_calls]
assert match_trajectory_tools(tools_called, ["python_interpreter"], mode="superset")
```

Modes:
- `"superset"`: All expected tools must have been called (`actual >= expected`).
- `"subset"`: Only allowed tools were called (`actual <= expected`).
- `"strict"`: Exact sequence match (`actual == expected`).

### 2. System One Decision Model Evaluations

Replaces brittle, unstructured prompt-based judges with calibrated, typed System One Decision Models:

```python
from genai_tk.core.factories import get_decision_model
from genai_tk.core.decision_models.evaluators import (
    evaluate_correctness,
    evaluate_conciseness,
    evaluate_groundedness,
    evaluate_tool_selection,
)

decision_model = get_decision_model("clef_flash@openrouter")  # or "default", "fake"

# Correctness -> Calibrated NoulAnswer (noul in [0.0, 1.0])
verdict = evaluate_correctness(
    decision_model,
    question="Calculate 10th Fibonacci number",
    gold_answer="55",
    agent_answer="The 10th Fibonacci number is 55.",
)
print("Correctness probability:", verdict.noul)

# Conciseness -> Ordinal ScoreAnswer (0=padded, 1=acceptable, 2=concise)
conciseness = evaluate_conciseness(
    decision_model,
    question="What is 2+2?",
    agent_answer="4",
)
print("Conciseness score:", conciseness.score)

# Tool Selection -> ChoiceAnswer (optimal | suboptimal | incorrect)
tool_verdict = evaluate_tool_selection(
    decision_model,
    task="Calculate 15 * 3",
    available_tools=["calculator", "web_search"],
    selected_tools=["calculator"],
)
print("Tool selection quality:", tool_verdict.choice)
```

## Relationship to monitoring & Langfuse OTLP

NeMo Relay acts as the central telemetry and trajectory core:

- **Local ATOF Trajectory Store** (`data/trajectories/<run_id>/events.jsonl`): Structured, agent-readable source of truth for `cli trajectory` and offline evaluation.
- **Native OTLP Exporter to Langfuse**: NeMo Relay's native Rust/C++ `OpenTelemetrySubscriber` translates scopes to OpenInference conventions and streams binary protobuf OTLP directly to Langfuse (`http://localhost:3000/api/public/otel/v1/traces`).
- **Telemetry Verification**: Use `cli monitoring test` to verify end-to-end connectivity across Relay and Langfuse.
- **Trace Deep Linking**: Use `cli trajectory open <run_id> --backend langfuse` or `cli trajectory link <run_id>` to seamlessly jump from a local trajectory to the remote Langfuse trace.

## ATOF event shape (illustrative)

Scope start (LLM call):

```json
{"kind":"scope","scope_category":"start","atof_version":"0.1","uuid":"...","parent_uuid":"...","timestamp":"2026-08-21T12:00:00Z","name":"gpt-oss-120b","category":"llm","category_profile":{"model_name":"gpt-oss-120b"},"attributes":["streaming"]}
```

LLM end (carries the annotated response — model, usage, tool calls):

```json
{"kind":"scope","scope_category":"end","category":"llm","category_profile":{"annotated_response":{"model":"gpt-oss-120b","usage":{"prompt_tokens":5542,"completion_tokens":62},"tool_calls":[{"name":"echo","arguments":{"message":"hello"}}]}}}
```

Skill-load mark (emitted automatically when an instrumented tool reads a `SKILL.md`):

```json
{"kind":"mark","name":"skill.load","data":{"skill_name":"navigation"},"metadata":{"skill_load_source":"structured_read","tool_name":"read_file"}}
```

## See also

- [docs/monitoring.md](monitoring.md) — multi-backend tracing (LangSmith / LangFuse / OTEL)
- [docs/evaluation_testing.md](evaluation_testing.md) — the eval test framework
- [docs/design/agent_trajectory_nemo_relay.md](design/agent_trajectory_nemo_relay.md) — design memo and next phases
- [docs/agents.md](agents.md) — agent frameworks and the harness layer
