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

The read layer is `genai_tk.utils.trajectory_store.TrajectoryStore`, which
parses ATOF events into typed `Trajectory` / `LlmCall` / `ToolCall` /
`SkillLoad` objects and projects a run to OpenAI-format messages.

## CLI: `cli trajectory`

| Command | Purpose |
|---|---|
| `cli trajectory list [--profile P] [--since WHEN] [--status failed]` | List recorded runs (id, profile, model, started, LLM/tool counts, tokens, status). |
| `cli trajectory show <id> [--format tree\|json\|messages\|dot]` | Render a trajectory. `tree` = scope timeline; `messages` = OpenAI-format; `dot` = scope-tree graph. |
| `cli trajectory tail [--n 20]` | Last N ATOF events from the most recent run. |
| `cli trajectory replay <id> [--delay 0.5]` | Replay events in order with relative timings. |
| `cli trajectory export <id> --format atif\|atof\|messages\|otel [--out file]` | Export a trajectory. |
| `cli trajectory diff <id1> <id2>` | Structural diff (tools, skills, step counts, tokens). |
| `cli trajectory skills <id>` | Show `skill.load` marks and where they occurred. |
| `cli trajectory stats [--since WHEN]` | Aggregate: token totals, tool/skill frequency, failure rate, latency p50/p95. |
| `cli trajectory prune [--keep-last N] [--older-than DAYS]` | Retention. |
| `cli trajectory view` | Launch the [Harbor](https://www.harborframework.com/) ATIF web viewer on the store (no-op if `harbor` isn't installed). |

`show --format messages` and `export --format messages` are the bridge to the
`agentevals` / `openevals` evaluation stack — the same OpenAI-format message
list, but read from a **real captured trajectory** instead of re-running the
agent.

```bash
# List recent runs
uv run cli trajectory list

# Inspect one run as a scope timeline
uv run cli trajectory show <run_id>

# Export the captured trajectory as OpenAI messages for offline eval
uv run cli trajectory export <run_id> --format messages --out run.json

# See which skills were loaded
uv run cli trajectory skills <run_id>

# Launch the Harbor web viewer (uv tool install harbor)
uv run cli trajectory view
```

## Programmatic access

```python
from genai_tk.utils.trajectory_store import TrajectoryStore

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

## Evals reading from the store

Store-based evaluation loads a **captured** trajectory and judges it — no agent
re-run. See `genai_tk.agents.langchain.trajectory_store_io`:

```python
from genai_tk.agents.langchain.trajectory_store_io import (
    load_trajectory_messages,
    compare_trajectory_to_golden,
    judge_trajectory,
)

# Load the captured trajectory as OpenAI messages
messages = load_trajectory_messages("<run_id>")

# Structural comparison against a golden reference
verdict = compare_trajectory_to_golden("<run_id>", {"tools": ["echo"], "min_steps": 4})
assert verdict["pass"]

# Run judges over the captured trajectory
verdicts = judge_trajectory(
    "<run_id>",
    [
        {"kind": "tool_use", "tools": ["echo"]},
        {"kind": "grounding"},
        {"kind": "efficiency", "max_repeat": 3},
        {"kind": "correctness", "judge": judge_llm, "reference_outputs": "echo:hello"},
    ],
)
```

Judge kinds:

| Kind | LLM? | What it checks |
|---|---|---|
| `correctness` | yes (`openevals`) | Final answer matches the reference output. |
| `trajectory_accuracy` | yes (`agentevals`) | Tool-call trajectory quality vs a reference trajectory. |
| `tool_use` | no | Expected tools ⊆ actual tools called. |
| `grounding` | no | Assistant answers follow a tool observation. |
| `efficiency` | no | No tool called more than `max_repeat` times. |

The deterministic judges (`tool_use` / `grounding` / `efficiency`) need no API
key and run in the default eval suite; the LLM judges are gated behind
`--include-real-models`.

## Relationship to monitoring

The legacy multi-backend tracing (`docs/monitoring.md` — LangSmith, LangFuse,
OTEL, local JSONL) and the trajectory store are complementary:

- **Trajectory store** = the local, structured, agent-readable record
  (ATOF scopes, tool args/results, skill loads, token usage). Source of truth
  for `cli trajectory` and store-based evals.
- **Remote backends** = dashboards and long-term retention (LangFuse, Phoenix,
  LangSmith). These are projections of the same event stream.

The monitoring bootstrap (`setup_monitoring()`) activates the Relay ATOF
subscriber alongside the configured remote backends, so a single agent run is
captured once and fanned out to all sinks.

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
