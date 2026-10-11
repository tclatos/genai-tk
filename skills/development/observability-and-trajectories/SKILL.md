---
name: observability-and-trajectories
description: Use and configure NeMo Relay OTLP telemetry, Langfuse export, ATOF TrajectoryStore, and terminal trajectory replay (cli trajectory tui) in genai-tk. Use when working on agent tracing, replay, or evaluation.
---

# Observability, Trajectories & Evaluation

## Read First

- `docs/design/09_observability_and_trajectories.md`
- `docs/trajectory.md`
- `docs/monitoring.md`
- `genai_tk/extra/monitoring/nemo_relay_setup.py`
- `genai_tk/extra/monitoring/trajectory_store.py`
- `genai_tk/cli/commands_trajectory.py`

## Core Concepts

1. **NeMo Relay OTLP Tracing**:
   Streams OpenTelemetry (OTLP) traces directly to Langfuse and LangSmith. Configured via environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
2. **Agent Trajectory Observability Format (ATOF)**:
   Structured JSONL representation of agent execution: thoughts, tool calls, tool results, tokens, latencies, and scopes.
3. **TrajectoryStore**:
   Disk-backed append-only trajectory repository (`genai_tk/extra/monitoring/trajectory_store.py`).
4. **Terminal TUI Replay**:
   Interactive terminal viewer (`cli trajectory tui`) for inspecting step-by-step cognitive traces without external browser dependencies.
5. **Decision Model Evaluators**:
   Calibrated System 1 scoring rubrics and deterministic tool sequence matching (`match_trajectory_tools`) replacing legacy regex LLM judges.

## Python Usage

```python
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore
from genai_tk.extra.monitoring.nemo_relay_setup import init_nemo_relay

# 1. Initialize telemetry
relay = init_nemo_relay(service_name="research_agent")

# 2. Access stored trajectories
store = TrajectoryStore(root="data/trajectories")
recent_trajs = store.list_trajectories(limit=10)

# 3. Deterministic tool verification in tests
latest = store.load_trajectory(recent_trajs[0].session_id)
assert latest.match_tools(["list_documents", "get_section_content"], mode="superset")
```

## CLI Verification

```bash
# Launch interactive terminal TUI
uv run cli trajectory tui

# List recent trajectories
uv run cli trajectory list --limit 5

# Replay single session
uv run cli trajectory show <session_id>
```
