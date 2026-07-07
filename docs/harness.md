# Harness Layer (`genai_tk.agents.harness`)

## Overview

genai-tk supports two agent runtimes: **LangChain** (react | deep | custom —
including the DeepAgents SDK) and **DeerFlow**. Both are built on
LangChain/LangGraph, so instead of maintaining two parallel CLI and UI code
paths, they share:

- one abstract session interface (`BaseHarness`)
- one Pydantic event model (`genai_tk.agents.harness.events`)
- one profile registry (`create_harness()` / `list_harness_profiles()`)
- the same middleware classes (no harness-specific middleware wrapper)

This is a thin normalization boundary, not a second agent framework — each
harness adapter still delegates all actual agent logic to its underlying
runtime (`create_langchain_agent()` / `EmbeddedDeerFlowClient`).

## Why not SmolAgents or a `deepagent` CLI?

SmolAgents and the never-implemented `deepagent-cli` bridge have been
removed. DeepAgents is a **LangChain agent type** (`type: deep` in a unified
`agents:` profile with `harness: langchain`, backed by the `deepagents` SDK) —
it does not need a separate command group or harness adapter; `LangChainHarness`
already handles it via the same `create_langchain_agent()` factory used for
`react` and `custom` profiles.

## Core Types

### Events (`genai_tk.agents.harness.events`)

All events are Pydantic models (not dataclasses), each a `Literal["kind"]`
discriminated variant of the `StreamEvent` union:

| Event | Fields | Meaning |
|---|---|---|
| `TokenEvent` | `text` | Incremental or complete assistant text |
| `NodeEvent` | `node`, `state` | A graph phase became active (planner, researcher, …) |
| `ToolCallEvent` | `tool_name`, `args`, `call_id` | The model is calling a tool |
| `ToolResultEvent` | `tool_name`, `content`, `call_id` | A tool returned a result |
| `ArtifactEvent` | `type`, `title`, `content`, `language` | A renderable artifact |
| `ClarificationEvent` | `question`, `clarification_type`, `context` | Agent paused for human input (HITL) |
| `UsageEvent` | `input_tokens`, `output_tokens` | Token usage for the turn |
| `ErrorEvent` | `message` | The run produced an error |
| `EndEvent` | — | The run has completed |

### `BaseHarness` (abstract base class)

```python
from genai_tk.agents.harness import BaseHarness

class BaseHarness(ABC):
    name: str

    @abstractmethod
    def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]: ...

    async def arun(self, message: str, *, thread_id: str | None = None) -> str: ...       # concrete
    async def list_threads(self) -> list[HarnessThread]: ...                              # default: []
    async def list_models(self) -> list[HarnessModel]: ...                                # default: []
    async def list_skills(self) -> list[HarnessSkill]: ...                                # default: []
    async def aclose(self) -> None: ...                                                   # default: no-op
```

An abstract base class was chosen over a `Protocol` so that `arun()` and the
default introspection methods can be implemented once and inherited by both
adapters, rather than duplicated.

### Adapters

- **`LangChainHarness`** (`genai_tk.agents.harness.langchain_harness`) — wraps
  `create_langchain_agent()`. Uses LangGraph's `astream_events()` (version
  `"v2"`) so `react`, `deep` (DeepAgents SDK), and `custom` profiles all stream
  uniformly — they are all compiled LangGraph graphs under the hood.
- **`DeerFlowHarness`** (`genai_tk.agents.harness.deerflow_harness`) — wraps
  `EmbeddedDeerFlowClient`. Translates DeerFlow's own event dataclasses
  (`TokenEvent`, `NodeEvent`, `ToolCallEvent`, …, defined in
  `genai_tk.agents.deer_flow.embedded_client`) into the canonical harness
  events. DeerFlow's own dataclasses are unchanged internally — the
  translation happens only at the harness boundary — so the unified
  `cli agents run` command and the DeerFlow Streamlit page's lower-level
  helpers keep working as-is.

### Registry (`genai_tk.agents.harness.registry`)

```python
from genai_tk.agents.harness import create_harness, list_harness_profiles

harness = create_harness("research")     # single dict lookup across all profiles
refs = list_harness_profiles()           # one combined list
```

`create_harness(key)` resolves `key` once against a single unified profile
dict returned by
:func:`~genai_tk.agents.harness.profiles.load_agent_profiles`, doing a
case-insensitive lookup that matches either the profile's dict key (slug) or
its `name`. It never probes two config trees separately any more.

**Unified profile model.** Both profile variants are exposed through the
:data:`~genai_tk.agents.harness.AgentProfile` discriminated union, keyed by
the `harness` field on each profile model:

```python
# genai_tk/agents/langchain/config.py
class AgentProfileConfig(BaseModel):
    harness: Literal["langchain"] = "langchain"
    ...

# genai_tk/agents/deer_flow/profile.py
class DeerFlowProfile(BaseModel):
    harness: Literal["deerflow"] = "deerflow"
    ...
```

**Canonical source.** Profiles live in a single `agents:` dict, resolved from
a project-level `config/agents.yaml` / `config/agents/` directory or the bundled
`config/examples/agents/` directory, with a top-level optional `agent_defaults:`
block for langchain inheritable defaults. The legacy split form
(`langchain_agents:` dict + `deerflow_agents:` list) is no longer supported.

## CLI

```bash
uv run cli agents list                                     # profiles from both config trees
uv run cli agents run research "Summarize recent AI safety news"
uv run cli agents run "Web Browser" "Go to atos.net" --llm gpt_41mini@openai
uv run cli agents run research "..." --json                # raw NDJSON events
```

Cross-harness flags on `run` cover framework-specific behaviour:

```bash
uv run cli agents run research --chat --sandbox docker           # DeerFlow sandbox
uv run cli agents run "Research Assistant" --mode ultra --trace  # DeerFlow mode + trace
```

## Middleware Is Shared, Not Adapted

DeerFlow's embedded client already forwards LangChain `AgentMiddleware`
instances to the upstream `DeerFlowClient` (used today for
`RichToolCallMiddleware`). Because of this, `AnonymizationMiddleware` and
`SensitivityRouterMiddleware` run **unmodified** in DeerFlow profiles — no
harness-neutral middleware vocabulary (`ModelIoMiddleware`,
`ToolPolicyMiddleware`, etc.) is needed. `DeerFlowProfile.middlewares` uses the
exact same `class` + kwargs shape as LangChain profiles and is instantiated
through the same `instantiate_middlewares()` factory (which also resolves any
`model`/`safe_llm` kwarg via `LlmFactory`). See
[middleware-pii-and-routing.md](middleware-pii-and-routing.md#cross-harness-usage-deerflow).

## Streamlit Workbench

The unified demo page (`genai_tk/webapp/pages/demos/agent.py`) renders through
`genai_tk.webapp.ui_components.harness_workbench`:

- `Artifact`, `ToolDetail`, `TraceStep`, `TurnResult` — shared Pydantic models
- `stream_harness_turn(harness, message, thread_id=..., response_placeholder=...)`
  — drives any `BaseHarness`, building trace steps and artifacts from the
  canonical event stream
- `render_trace_panel()`, `render_artifact()`, `render_artifact_gallery()` —
  shared rendering functions

The page uses a two-column layout: execution trace (left) + chat/artifact
tabs (right). An `st.pills` filter narrows profiles by kind (React, DeepAgent,
Custom, DeerFlow) and a profile selector picks one across both harnesses.

## Adding a New Harness

1. Create `genai_tk/agents/harness/<name>_harness.py` with a class extending
   `BaseHarness`, implementing `astream()` and translating the runtime's
   native events into `genai_tk.agents.harness.events` types.
2. Add a profile model with a `harness: Literal["<name>"]` discriminator
   field (or reuse an existing one if the runtime already has one).
3. Wire it into `create_harness()` / `list_harness_profiles()` in
   `registry.py`.
4. No changes needed in the CLI or Streamlit workbench — they only depend on
   `BaseHarness` and the canonical events.

## See Also

- [agents.md](agents.md) — full agent configuration reference
- [deer-flow.md](deer-flow.md) — DeerFlow-specific setup and profile fields
- [middleware-pii-and-routing.md](middleware-pii-and-routing.md) — anonymization + routing middleware
- [monitoring.md](monitoring.md) — harness trace metadata and project naming
- [webapp.md](webapp.md) — Streamlit demo pages
- `notebooks/harness_quickstart.ipynb`, `notebooks/harness_middleware_demo.ipynb`

## DeepAgents ↔ DeerFlow Event Parity

Both adapters emit the same canonical event vocabulary, including `NodeEvent`
(graph phase became active). DeerFlow emits planner/researcher/coder/reporter
phase events natively; `LangChainHarness` derives `NodeEvent`s from LangGraph's
`on_chain_start` events for `deep` (DeepAgents SDK) profiles — internal
plumbing node names (`LangGraph`, `agent`, `tools`, `model`,
`should_continue`) and the root graph invocation are filtered out so only
meaningful phases surface in the Streamlit workbench's left rail.

`ArtifactEvent` is currently only emitted by `DeerFlowHarness`; DeepAgents
artifacts remain inside `ToolResultEvent`s until a richer shared detection is
added (a deferred, workbench-renderer concern).

## Trace Metadata

Both harness adapters set canonical trace metadata at startup via
`apply_harness_trace_metadata(...)` — see [monitoring.md](monitoring.md#harness-trace-metadata).
Trace project names follow `GenAITk-<harness>-<profile>` for both runtimes.
