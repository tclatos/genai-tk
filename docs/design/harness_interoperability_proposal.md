# Harness Interoperability Proposal

## Goal

Align DeerFlow, Deep Agents, and the existing LangChain agent stack behind a
small set of shared genai-tk concepts:

- one model resolution path
- one skills and MCP story
- one sandbox story
- one monitoring story
- one reusable UI event model

The intent is not to hide the differences between harnesses. The intent is to
make those differences explicit and isolate them behind thin adapters, so the
toolkit can support multiple harnesses without forking its architecture.

---

## Executive Summary

The current genai-tk already has most of the right building blocks:

- `LlmFactory` for model resolution
- shared skills directories
- MCP server configuration
- a reusable Docker sandbox backend for deepagents-style execution
- a multi-backend monitoring layer with LangSmith, LangFuse, OTEL, and local logs
- an embedded DeerFlow client and a dedicated Streamlit page

What is missing is a shared harness layer.

Today:

- DeerFlow is exposed as a first-class harness with its own CLI and Streamlit page.
- LangChain deep agents exist, but only inside `cli agents langchain`.
- There is no source implementation of the documented standalone deepagent CLI bridge.
- Upstream Deep Agents moved the interactive TUI from `deepagents-cli` to `deepagents-code` (`dcode`).
- Local docs still mix old and new DeerFlow setup models.

The recommendation is to standardize genai-tk around a `harness` adapter layer,
then add thin command and UI wrappers for DeerFlow and Deep Agents Code on top.

---

## Current State In genai-tk

### What exists now

1. DeerFlow is the most complete harness integration.

   - `genai_tk.agents.deer_flow.embedded_client.EmbeddedDeerFlowClient` wraps the upstream embedded client pattern.
   - `genai_tk.agents.deer_flow.cli_commands` provides `cli agents deerflow`.
   - `genai_tk.webapp.pages.demos.deer_flow_agent` provides a dedicated Streamlit workbench.

2. LangChain already acts as a generic agent substrate.

   - `cli agents langchain` supports `react`, `deep`, and `custom` profiles.
   - The deep profile path already reuses deepagents SDK primitives.
   - LangChain middleware is already a real extension point.

3. Monitoring is already more advanced in genai-tk than in most harnesses.

   - `genai_tk.utils.tracing` supports LangSmith, LangFuse, OTEL, and local JSONL.
   - The DeerFlow bridge already sets trace metadata and trace project names.

4. Shared UI pieces exist, but only at layout level.

   - `genai_tk.webapp.ui_components.agent_layout` is shared.
   - Event capture and rendering are still harness-specific.

### What does not exist now

1. No standalone Deep Agents CLI bridge is present in source.

   The design document in `docs/design/deepagents-cli_integration.md` describes a
   `genai_tk.agents.deepagent_cli` package, but that package is not present in
   the repository.

2. No shared harness event protocol exists.

   DeerFlow streams node and tool events. LangChain pages rely on callback
   middleware. SmolAgents has its own rendering helpers. None of these are yet
   normalized into one UI/event abstraction.

3. The docs have drifted.

   DeerFlow docs now describe an embedded harness install, but parts of the
   README still refer to cloning DeerFlow and setting `DEER_FLOW_PATH`.

---

## Relevant Upstream Changes

### DeerFlow 2.0

The new DeerFlow release is important for genai-tk because it is now much more
obviously a general harness rather than only a research workflow.

Relevant upstream features:

1. Native terminal workbench.

   DeerFlow now ships a TUI launched by `deerflow`, backed by the embedded
   `DeerFlowClient`. It supports:

   - thread resume and switching
   - model switching
   - slash commands
   - headless `--print` and `--json` modes
   - shared persistence so TUI sessions appear in the Web UI

2. Built-in LangSmith and LangFuse tracing.

   DeerFlow supports both providers at once and injects stable trace metadata:

   - session id from thread id
   - user id
   - assistant id / trace name
   - model and environment tags

3. Embedded client parity.

   The embedded Python client is no longer a side feature. It exposes a large
   management surface, including streaming, models, skills, MCP, memory,
   threads, uploads, and artifacts.

4. Web UI and terminal UI share thread persistence.

   This is especially relevant to genai-tk because we already have a Streamlit
   UI and should avoid building isolated per-harness histories.

### Deep Agents / Deep Agents Code

The important upstream change is naming and packaging.

1. `deepagents-cli` is no longer the interactive coding agent.

   It is now deployment tooling.

2. The interactive TUI moved to `deepagents-code` (`dcode`).

   Relevant features:

   - interactive terminal UI
   - model switcher
   - agent switching
   - threads browser and resume
   - headless mode for scripting and CI
   - shell allow-list model
   - persistent memory and skills
   - MCP tool support
   - LangSmith tracing with dedicated project controls

3. Deep Agents SDK remains the harness substrate.

   It still provides the deeper runtime concepts that genai-tk cares about:

   - planning
   - sub-agents
   - filesystem abstraction
   - skills
   - tool middleware
   - sandbox backends

This means genai-tk should distinguish three different things:

- Deep Agents SDK: the programmable harness
- Deep Agents Code: the packaged TUI and headless coding tool
- deepagents-cli: deployment tooling, not the TUI surface

---

## Architectural Comparison

| Area | DeerFlow | Deep Agents SDK / Code | Current genai-tk position |
|---|---|---|---|
| Core runtime | LangGraph harness with embedded client and gateway parity | Opinionated harness on top of LangGraph / LangChain | Already supports both LangGraph-style and LangChain-style runtimes |
| TUI | Native `deerflow` workbench | Native `dcode` TUI | No unified wrapper surface yet |
| Headless mode | `--print`, `--json` | `-n`, piping, quiet output | Could normalize behind toolkit commands |
| Skills | Progressive `SKILL.md` loading | Progressive skill loading | Already a toolkit strength |
| MCP | First-class | First-class | Already shared in config |
| Sandbox | local, Docker, provisioner/K8s | local or remote sandbox backends | Toolkit already has shared sandbox building blocks |
| Monitoring | LangSmith + LangFuse | LangSmith-first | Toolkit already has multi-backend tracing |
| Web UI | Native web UI plus shared thread store | No equivalent built into SDK; TUI is primary | Toolkit Streamlit UI can fill this gap |

The common denominator is clear enough to standardize on.

---

## Key Problems To Solve

### 1. Command surface fragmentation

The command model is currently inconsistent:

- `cli agents deerflow` is a dedicated harness command.
- `cli agents langchain` includes both ReAct and deepagents SDK profiles.
- no equivalent dedicated command exists for Deep Agents Code.

This makes feature discovery uneven and hides the fact that DeerFlow and Deep
Agents are peers.

### 2. Middleware is not yet harness-agnostic

The user requirement that anonymization should run in DeerFlow or LangChain /
DeepAgents is not guaranteed by the current architecture.

Today middleware is mostly attached at harness-specific extension points.

That is too low-level for cross-harness policies such as:

- anonymization
- prompt redaction
- policy tags
- trace metadata enrichment
- tool audit logging

### 3. Monitoring semantics are inconsistent across harnesses

genai-tk already supports multiple backends, but harnesses annotate and expose
runs differently.

The toolkit needs one canonical notion of:

- trace project naming
- session id / thread id
- user id
- assistant or profile name
- environment tags
- trace URL discovery

### 4. Streamlit UI is layout-shared, not runtime-shared

The current Streamlit pages demonstrate the right UX direction, but not yet the
right architecture.

The DeerFlow page is event-driven and artifact-aware. The ReAct page is callback
oriented. These should converge into a shared workbench model.

---

## Proposal

### 1. Introduce a Harness Adapter Layer

Add a new internal package, for example:

```text
genai_tk/agents/harness/
  base.py
  events.py
  registry.py
  deerflow_adapter.py
  langchain_adapter.py
  dcode_adapter.py
```

Core protocol:

```python
class HarnessSession(Protocol):
    async def stream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[HarnessEvent]: ...
    async def list_threads(self) -> list[HarnessThread]: ...
    async def list_models(self) -> list[HarnessModel]: ...
    async def list_skills(self) -> list[HarnessSkill]: ...
```

Canonical event types:

- `token`
- `status`
- `tool_call`
- `tool_result`
- `artifact`
- `clarification`
- `trace`
- `usage`
- `end`
- `error`

This adapter layer should stay thin. It is not a second agent framework. It is
only a normalization boundary for CLI and UI features.

### 2. Standardize command families

Target command model:

```bash
cli agents langchain      # generic LangChain agents, including react/custom
cli agents deerflow       # DeerFlow harness wrapper
cli agents deepagent      # toolkit wrapper for Deep Agents Code / SDK
```

Recommended behavior:

1. `cli agents deepagent` should be a toolkit-managed surface, even if the
   implementation delegates to `dcode` or to SDK adapters.

2. Common flags should be aligned as far as possible:

   - `--profile, -p`
   - `--llm, -m`
   - `--resume`
   - `--trace`
   - `--sandbox`
   - `--json`
   - `--print` or headless one-shot mode

3. Harness-specific flags should still exist, but only after a common baseline.

This keeps the toolkit opinionated without pretending the harnesses are identical.

### 3. Separate toolkit middleware from harness middleware

Introduce a harness-neutral middleware vocabulary at the toolkit level.

Suggested split:

1. `ModelIoMiddleware`

   Operates on prompts, intermediate model messages, and final answers.

   Use for:

   - anonymization
   - de-anonymization
   - redaction
   - system metadata injection

2. `ToolPolicyMiddleware`

   Operates on tool requests and tool results.

   Use for:

   - allow/deny policy
   - audit logging
   - argument scrubbing
   - result truncation

3. `TraceMetadataMiddleware`

   Operates on per-run metadata.

   Use for:

   - session id
   - user id
   - profile name
   - model name
   - environment tags

Then each harness adapter maps these toolkit middlewares to the harness-native
extension points.

This is the cleanest path to the anonymizer requirement.

### 4. Make monitoring semantics first-class and shared

Adopt DeerFlow's trace metadata discipline across the toolkit.

Canonical metadata fields should include:

- `thread_id`
- `session_id`
- `user_id`
- `assistant_id` or `profile_name`
- `model_name`
- `harness`
- `environment`

Also adopt one toolkit-level trace naming convention, for example:

- `GenAITk-DeerFlow-<profile>`
- `GenAITk-DeepAgent-<profile>`
- `GenAITk-LangChain-<profile>`

Deep Agents Code has a useful idea here: separate the toolkit's own traces from
the app traces it launches in subprocesses or shells. genai-tk should adopt the
same distinction when running tests, scripts, or nested agent systems.

### 5. Build one reusable Streamlit workbench

Instead of one page per runtime style, build a shared workbench that consumes
the new `HarnessEvent` stream.

Suggested structure:

- left rail: execution phases, tool calls, usage, trace link
- center/right tab 1: transcript
- center/right tab 2: artifacts
- center/right tab 3: thread or memory info

The current DeerFlow page is the best starting point because it already has:

- phase cards
- artifact extraction
- multi-turn thread state

The LangChain ReAct page should be refactored to emit the same event model,
instead of driving the UI directly from callback-specific structures.

### 6. Reuse shared thread persistence where feasible

DeerFlow's shared TUI/Web persistence is a strong design pattern.

genai-tk should adopt the idea at toolkit level:

- a harness adapter can expose thread metadata
- the Streamlit workbench can list and resume threads across harnesses
- TUI and web should not create separate silos when the underlying harness allows sharing

For DeerFlow this can map closely to upstream behavior.

For Deep Agents Code this may start as read-only interoperability, then evolve
into richer sharing later.

---

## Recommended Near-Term Work

### Phase 0: Clean up naming and documentation

Do this first.

1. Update toolkit docs to reflect that:

   - DeerFlow no longer needs `DEER_FLOW_PATH` for the embedded harness path.
   - `deepagents-cli` is not the interactive TUI anymore.
   - the relevant interactive Deep Agents surface is `deepagents-code` (`dcode`).

2. Clarify the distinction between:

   - Deep Agents SDK
   - Deep Agents Code
   - deepagents-cli

3. Document the current state honestly: DeerFlow is integrated now; Deep Agents
   standalone TUI integration is a design target, not a completed feature.

### Phase 1: Add the harness adapter layer

This is the highest leverage implementation step.

Scope:

- create common event models
- implement DeerFlow adapter first
- implement LangChain adapter second
- use adapters only in Streamlit at first

This de-risks the UI convergence before touching the CLI.

### Phase 2: Add `cli agents deepagent`

Implement a toolkit wrapper that targets the current upstream reality.

Recommended order:

1. headless one-shot mode
2. resume / thread listing
3. TUI launch wrapper
4. model and profile mapping through `LlmFactory`

Important: this wrapper should be designed around `deepagents-code`, not the
older `deepagents-cli` TUI assumptions.

### Phase 3: Make the anonymizer harness-neutral

Move anonymization into the toolkit middleware split described above, then wire
it into:

- LangChain agents
- DeerFlow embedded client
- Deep Agents wrapper

### Phase 4: Replace per-harness Streamlit pages with a shared workbench

Retain harness-specific side controls, but standardize the central workbench.

---

## Features Worth Prioritizing From Upstream

These upstream features are immediately relevant to genai-tk.

1. DeerFlow TUI as a reference design for a terminal workbench.

   Especially relevant:

   - slash command palette
   - thread/model pickers
   - compact tool activity cards
   - shared persistence between terminal and web

2. DeerFlow trace metadata design.

   The session and user propagation model is cleaner than ad hoc per-command
   environment setup.

3. Deep Agents Code command ergonomics.

   Especially relevant:

   - headless mode for CI
   - shell allow-list model
   - `/trace`, `/threads`, `/model`
   - explicit separation between agent traces and application traces

4. DeerFlow embedded client breadth.

   Its management API is a strong example of how a harness can expose models,
   skills, threads, MCP, uploads, and memory through one coherent client.

---

## Risks And Constraints

1. Deep Agents interactive support has moved upstream.

   Any toolkit implementation that assumes the old deepagents-cli TUI model will
   age badly.

2. Harnesses are not perfectly isomorphic.

   DeerFlow has stronger native web and artifact concepts. Deep Agents Code has
   richer terminal-specific workflow concepts. The adapter layer should normalize
   only the overlap.

3. Middleware portability is easiest at the toolkit boundary, not by forcing all
   upstream harnesses to accept the same class types.

4. Thread-store interoperability may need to remain best-effort across harnesses.

---

## Recommendation

The toolkit should move toward a three-layer model:

1. Shared toolkit substrate

   - LLM resolution
   - skills
   - MCP
   - sandbox config
   - monitoring
   - toolkit middlewares

2. Harness adapters

   - DeerFlow adapter
   - LangChain adapter
   - Deep Agents adapter

3. Presentation surfaces

   - CLI command groups
   - TUI launch wrappers
   - one shared Streamlit workbench

This preserves the strengths of each harness while giving genai-tk one coherent
story for interoperability.

---

## Concrete Next Steps

1. Fix the docs drift around DeerFlow setup and Deep Agents packaging.
2. Introduce `genai_tk.agents.harness.events` and a DeerFlow adapter.
3. Refactor the current DeerFlow Streamlit page so its rendering consumes shared
   event objects rather than DeerFlow-specific state shapes.
4. Add a LangChain adapter that emits the same event objects from middleware.
5. Design `cli agents deepagent` around `deepagents-code`, not the old
   deepagents-cli TUI model.
6. Move the anonymizer into a harness-neutral toolkit middleware layer.