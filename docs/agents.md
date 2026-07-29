# Agents Module (`genai_tk.agents`)

## Overview

The `agents` module provides agent implementations including LangChain-based agents (ReAct, Deep — including the DeepAgents SDK —, Custom) and DeerFlow. All agents are configuration-driven with support for tools, MCP servers, middlewares, and checkpointing. A shared **harness layer** (`agents.harness`) normalizes both runtimes behind one event model and one CLI/UI surface — see [Harness Layer](#harness-layer-agentsharness) below.

## LangChain Agents (`agents.langchain`)

### Unified Architecture

The LangChain agent module uses a **unified configuration system** supporting three agent types:

- **ReAct** - Standard reasoning agent with tool use loops
- **Deep** - Advanced reasoning with planning and subagents (requires `deepagents` package; run via `cli agents run <profile>`)
- **Custom** - Functional API-based custom agent from scratch

All agent types are managed through a single configuration interface with sensible defaults and profile-based customization.

### Configuration System

Agent profiles live under one **unified `agents:` dict** — both LangChain (`react` | `deep` | `custom`) and DeerFlow profiles — discriminated by a `harness:` field on each profile. The dict key is the profile slug used to select it from the CLI (e.g., `cli agents run research`).

**Config directory:** `config/examples/agents/` — category files (`defaults.yaml`, `simple.yaml`, `deep.yaml`, `browser.yaml`, `text2sql.yaml`, `deerflow.yaml`) merged into one `agents:` dict.

**defaults.yaml** — inheritable defaults for `harness: langchain` profiles:
```yaml
agent_defaults:
  type: react                          # default: react | deep | custom
  llm: null                            # null = use default from config
  enable_planning: true                # for deep agents only
  enable_file_system: true             # for deep agents only
  middlewares:                         # default middleware pipeline
    - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
  checkpointer:
    type: none                         # none | memory | class
  backend:
    type: none                         # none | aio_sandbox | class
  skills:
    directories:                       # skill search directories
      - ${paths.project}/skills
  default_profile: "simple"            # profile selected when -p is omitted
```

DeerFlow profiles are fully self-describing and do **not** inherit from `agent_defaults`. When a LangChain profile omits `harness:`, it defaults to `langchain`.

**deep.yaml** — Deep agent profiles:
```yaml
agents:
  # Profile key: "research" (used as: cli agents run research)
  research:
    name: "Research"                     # Display name (for list output)
    type: deep
    llm: gpt_41@openai
    enable_planning: true
    tools:
      - spec: web_search
        config:
          provider: serper
    mcp_servers: []
      
  # Profile key: "coding"
  coding:
    name: "Coding"
    type: deep
    llm: gpt_4o@openai
    enable_file_system: true
    tools:
      - spec: filesystem_tools
    middlewares:
      - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
        details: true
      
  # Profile key: "data_analysis"
  data_analysis:
    name: "Data Analysis"
    type: react
    llm: gpt_4o@openai
    tools:
      - spec: sql_tools
        config:
          database: analytics.db
      - spec: dataframe_tools
```

**Note:** Profile **keys** (like `research`, `coding`) are lowercase and used in CLI commands. Profile **names** (like `Research`, `Coding`) are display names shown in `--list` output.

**Key Configuration Options:**

- `type` - Agent type: `react` | `deep` | `custom`
- `llm` - Model ID (e.g., `gpt_4o@openai`), null uses default
- `tools` - List of tool specifications with optional configs
- `mcp_servers` - MCP server names for dynamic tool loading
- `middlewares` - Middleware pipeline for tool calls and responses
- `checkpointer` - State persistence for multi-turn conversations
- `backend` - Execution backend (for deep agents)

### Agent Types

#### ReAct Agent (Default)

Standard reasoning agent using the ReAct pattern: Thought → Action → Observation loop.

**Best for:** General-purpose tasks, straightforward reasoning

**Configuration (config/examples/agents/simple.yaml):**
```yaml
agents:
  # Profile key: "simple"
  simple:
    name: "Simple"
    type: react
    llm: gpt_4o@openai
    tools:
      - spec: web_search
      - spec: calculator
```

**Usage:**
```python
from genai_tk.agents.harness.profiles import load_langchain_profiles
from genai_tk.agents.langchain.factory import create_langchain_agent

profile = load_langchain_profiles()["simple"]  # Profile KEY, not name
agent = await create_langchain_agent(profile)

# Single query
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}]
})

# Interactive chat (with memory)
result = await agent.ainvoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "user_123"}}
)
```

#### Deep Agent (Advanced)

Advanced reasoning agent with planning, subagents, and execution backends. Requires `deepagents` package.

**Features:**
- Multi-step planning and decomposition
- Subagent delegation
- Docker sandbox execution (optional)
- Enhanced error handling and recovery

**Best for:** Complex multi-step tasks, research, analysis

**Configuration (config/examples/agents/deep.yaml):**
```yaml
agents:
  # Profile key: "research"
  research:
    name: "Research"
    type: deep
    llm: gpt_41@openai
    enable_planning: true
    enable_file_system: true
    # Optional: Docker sandbox backend
    backend:
      type: aio_sandbox
      opensandbox_server_url: http://localhost:8080
      startup_timeout: 90.0
    skills:
      directories:
        - ${paths.project}/skills
    tools:
      - spec: web_search
      - spec: filesystem_tools
```

**Backend Types:**
- `none` - Standard execution
- `aio_sandbox` - Docker sandbox (requires opensandbox running)
- `class` - Custom backend class

**Note:** Deep agents require the `deepagents` library (optional dependency). There is no separate `deepagent` CLI — use `cli agents run <profile>` where the profile has `type: deep`.

#### Custom Agent

Functional API-based agent built with LangGraph's Functional API for maximum customization.

**Best for:** Specialized workflows, advanced graph topologies

**Configuration:**
```yaml
agents:
  custom:  # Profile key
    name: "Custom"
    type: custom
    llm: gpt_4o@openai
```

**Implementation Example:**
```python
from genai_tk.agents.langchain.factory import create_langchain_agent
from genai_tk.extra.graphs.custom_react_agent import create_custom_react_agent

# Framework automatically dispatches to create_custom_react_agent
agent = await create_langchain_agent(profile)
```

### Tools Configuration

Tools are loaded from the configuration with optional tool-specific settings:

```yaml
tools:
  - spec: web_search           # tool specification name
    config:                    # tool-specific configuration
      provider: serper
      max_results: 5
  - spec: calculator
  - spec: filesystem_tools
    config:
      allowed_dirs: ["/home", "/tmp"]
  - spec: sql_tools
    config:
      database: analytics.db
      schema: public
```

**Available Tool Specs:**
- `web_search` - Web search tool
- `calculator` - Mathematical calculator
- `filesystem_tools` - File system access
- `sql_tools` - SQL database tools
- `dataframe_tools` - Pandas/polars operations
- `python_repl` - Python code execution
- Custom specs via tool factory

### Middleware Pipeline

Middlewares enhance agent behavior with logging, rate limiting, and output formatting.

```yaml
middlewares:
  - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
    details: true                    # Show detailed output
  - class: deepagents.middleware.summarization.SummarizationMiddleware
    model: gpt-35-turbo@openai
    trigger: ["tokens", 4000]        # Summarize after 4000 tokens
```

**Built-in Middlewares:**
- `RichToolCallMiddleware` - Pretty-print tool calls with Rich
- `ToolCallLimitMiddleware` - Limit tool calls per thread
- `SummarizationMiddleware` - Summarize long conversations

### Checkpointing

Persist agent state for multi-turn conversations and recovery:

```yaml
checkpointer:
  type: memory              # memory | postgres | sqlite
  # PostgreSQL configuration:
  # type: postgres
  # connection_string: postgresql://user:pass@localhost/db
  # table_name: agent_checkpoints
```

**Thread-based State:**
```python
# Same thread_id maintains conversation history
result = await agent.ainvoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "user_session_123"}}
)

# Each call has access to previous conversation
result = await agent.ainvoke(  # Same thread_id
    {"messages": [{"role": "user", "content": "Continue..."}]},
    config={"configurable": {"thread_id": "user_session_123"}}
)
```

### MCP Servers Integration

Load tools from Model Context Protocol servers:

```yaml
mcp_servers:
  - math_server           # Named MCP server from config
  - weather_server
```

**Server Configuration:**
```yaml
mcp_servers_config:
  math_server:
    command: python
    args: ["-m", "math_server"]
    env:
      MCP_LOG_LEVEL: info
  weather_server:
    command: python
    args: ["-m", "weather_server"]
```

**Runtime Usage:**
```python
# Override MCP servers at runtime
agent = await create_langchain_agent(
    profile,
    extra_mcp_servers=["custom_server"]
)
```

### CLI Interface

One `cli agents run` command covers every profile (LangChain react/deep/custom
and DeerFlow); `cli agents list` lists them all.

**Interactive Shell:**
```bash
# Default profile with interactive chat
cli agents run --chat

# Specific profile (by KEY) with chat
cli agents run research --chat

# Single query with a specific profile
cli agents run coding "List Python files"
```

**Single-Shot Queries:**
```bash
# Default profile, query via stdin
echo "What is machine learning?" | cli agents run

# Override LLM
cli agents run research --llm gpt_4o@openai "Research AI"

# DeerFlow reasoning mode / sandbox overrides
cli agents run "Research Assistant" --mode ultra "Research AI"
cli agents run coding --sandbox docker "Refactor my code"

# List available profiles
cli agents list
```

**Shell Mode (programmatic):**
```python
from genai_tk.agents.langchain.agent_cli import run_langchain_agent_shell

# Start interactive agent shell for a LangchainAgent
await run_langchain_agent_shell(agent)
```

## SmolAgents removed

SmolAgents (`cli agents smol`) has been removed. Use `cli agents run <profile>`
for any agent — LangChain (react/deep/custom) and DeerFlow now share one
harness layer and one command (see below).

## Harness Layer (`agents.harness`)

Both LangChain and DeerFlow agents run on LangChain/LangGraph, so they share
one set of Pydantic event types and one abstract session interface instead of
maintaining parallel CLI/UI code paths per framework.

**Core types** (`genai_tk.agents.harness`):

```python
from genai_tk.agents.harness import BaseHarness, TokenEvent, ToolCallEvent, create_harness

harness = create_harness("research")   # resolves "research" across all
                                        # harnesses via load_agent_profiles()
async for event in harness.astream("What is RAG?"):
    if isinstance(event, TokenEvent):
        print(event.text, end="", flush=True)
```

`BaseHarness` (abstract base class, not a `Protocol`) defines:

| Method | Purpose |
|---|---|
| `astream(message, thread_id=None)` | Abstract — stream canonical `StreamEvent`s |
| `arun(message, thread_id=None)` | Concrete — consumes the stream, returns concatenated text |
| `list_threads()` / `list_models()` / `list_skills()` | Optional harness introspection |
| `aclose()` | Release resources (sandbox containers, connections) |
| `get_graph()` / `get_checkpointer()` | Compiled LangGraph graph / checkpointer, for introspection |

Event kinds (`genai_tk.agents.harness.events`): `TokenEvent`, `NodeEvent`,
`ToolCallEvent`, `ToolResultEvent`, `ArtifactEvent`, `ClarificationEvent`,
`UsageEvent`, `ErrorEvent`, `EndEvent`.

**Adapters:**

- `LangChainHarness` — wraps `create_langchain_agent()`; works for `react`,
  `deep` (DeepAgents SDK), and `custom` profiles via LangGraph's
  `astream_events()`. `get_graph()` returns the real, production graph.
- `DeerFlowHarness` — wraps `EmbeddedDeerFlowClient`, which now yields the
  canonical harness events directly (no separate translation step).
  `get_graph()` is best-effort/introspection-only — DeerFlow's real graph
  construction is tightly coupled to private tracing/authorization setup, so
  the accessor never drives production streaming; see
  [harness.md](harness.md#baseharness-abstract-base-class) for details.

**Profile discriminator:** every profile carries an explicit `harness` field
(`AgentProfileConfig.harness` = `"langchain"`, `DeerFlowProfile.harness` =
`"deerflow"`). `create_harness(key)` and `list_harness_profiles()` look the key
up in one unified profile dict (loaded by `load_agent_profiles()`) and dispatch
to the matching adapter — no YAML changes needed.

**CLI:** `cli agents run <profile> "<query>"` and `cli agents list` are the
single entry points and work across both harnesses. Cross-harness flags on
`run` (`--chat`, `--mode`, `--sandbox`, `--mcp`) cover framework-specific
behaviour — see [cli.md](cli.md).

**Middleware is shared, not adapted.** Since DeerFlow's embedded client
already forwards LangChain `AgentMiddleware` instances to the underlying
`DeerFlowClient`, the exact same `AnonymizationMiddleware` and
`SensitivityRouterMiddleware` classes run unmodified in DeerFlow profiles —
see [middleware-pii-and-routing.md](middleware-pii-and-routing.md).

**Streamlit:** the unified `🤖 Agent` demo page (`agent.py`) renders through
the shared `genai_tk.webapp.ui_components.harness_workbench` module (trace
phase cards, chat transcript, artifact gallery) driven by the same event
stream — see [webapp.md](webapp.md).

## Common Patterns

### Pattern 1: Profile-Based Agent Selection

```python
from genai_tk.agents.harness.profiles import load_langchain_profiles
from genai_tk.agents.langchain.factory import create_langchain_agent

# Load all LangChain profiles (agent_defaults already applied)
profiles = load_langchain_profiles()

# Select profile (by KEY, e.g., from CLI, environment, or hardcoded)
profile_key = "research"  # or get from args
profile = profiles[profile_key]

# Create and use agent
agent = await create_langchain_agent(profile)
result = await agent.ainvoke({"messages": [...]})
```

### Pattern 2: Runtime Tool Addition

```python
from langchain_core.tools import tool
import asyncio

@tool
def custom_tool(arg: str) -> str:
    """Custom tool description."""
    return f"Custom result: {arg}"

# Create agent and add tool
agent = await create_langchain_agent(profile, extra_tools=[custom_tool])
```

### Pattern 3: Checkpointed Conversations

```python
# Create agent with memory checkpointer
profile.checkpointer = CheckpointerConfig(type="memory")
agent = await create_langchain_agent(profile, force_memory_checkpointer=True)

# Multi-turn conversation
thread_id = "user_session_123"

# Turn 1
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Tell me about AI"}]},
    config={"configurable": {"thread_id": thread_id}}
)
print(result["messages"][-1].content)

# Turn 2 - context preserved
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What about ML?"}]},
    config={"configurable": {"thread_id": thread_id}}
)
```

### Pattern 4: Agent with Custom Backend

```python
from genai_tk.agents.langchain.config import BackendConfig

# Configure sandbox backend
backend = BackendConfig(
    type="aio_sandbox",
    opensandbox_server_url="http://localhost:8080",
    startup_timeout=90.0
)

# Use with deep agent
profile.type = "deep"
profile.backend = backend
agent = await create_langchain_agent(profile)
```

## Configuration

Agent profiles live under one unified `agents:` dict, resolved from a project
`config/agents.yaml` (or `config/agents/` directory), falling back to the
bundled `config/examples/agents/` category files.
See [docs/configuration.md](configuration.md) for the full configuration reference including environments, `.env` loading, and how to add new profiles.

## Debugging

**Enable Verbose Output:**
```python
from loguru import logger
logger.enable("genai_tk")

agent = await create_langchain_agent(profile, details=True)
```

**Trace Tool Calls:**
```bash
cli agents run research "Your query" --trace
```

**Inspect Configuration:**
```python
from genai_tk.agents.harness.profiles import load_langchain_profiles

profiles = load_langchain_profiles()
print(list(profiles.keys()))
print(profiles["research"].model_dump_json(indent=2))
```

## Testing LangchainAgent

The integration test suite is in
`tests/integration_tests/agents/test_langchain_agent_real.py` and covers four
areas:

| Area | What is tested | Requires |
|---|---|---|
| React agents | Code generation, Q&A, streaming, multi-turn memory | `--include-real-models` |
| Deep agents (local) | Code generation, file writes via `FilesystemBackend` | `--include-real-models` |
| Dict-keyed profiles | Field types, `enable_*` flags, profile resolution by key | (structural — no LLM) |
| Skills loading | SKILL.md discovery, backend wiring, content access | `--include-real-models` |
| Docker sandbox | Full container run, file writes in container | `--include-real-models --include-docker` |

### Running the tests

```bash
# Fast structural checks (no LLM, no API keys)
uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \
    -k "profile_is or profile_loads or skill_directory_resolves" -v

# All agent tests with real model
uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \
    -v --include-real-models --timeout=180

# Include Docker sandbox tests (requires Docker daemon)
uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \
    -v --include-real-models --include-docker
```

### Writing new agent tests

The file provides a small helper set to keep new tests concise:

```python
from tests.integration_tests.agents.test_langchain_agent_real import _run, _has, LLM

@pytest.mark.integration
@pytest.mark.real_models
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_my_new_agent() -> None:
    agent = LangchainAgent(llm=LLM, agent_type="react")
    result = await _run(agent, "Your query here")
    assert _has(result, "expected", "keyword")
```

Key patterns:

- Use `_run(agent, query)` — creates the agent, runs it, and always closes it.
- Use `_has(text, *words)` — case-insensitive substring check, returns `True` if any word matches.
- Use `LLM = "fast_model"` throughout — resolves to `claude-haiku@openrouter` (cheap and reliable).
- Use `pytest.importorskip("deepagents")` in deep-agent tests — auto-skip if package is missing.
- Mark flaky or environment-dependent tests with `pytest.xfail(...)` rather than removing them.
- Use `async with agent: ...` (or `_run`) to guarantee `close()` is called on every code path.

### Test tiers

```
tests/integration_tests/agents/
├── test_langchain_agent_integration.py  — basic lifecycle (fake + 1 real-model test)
├── test_langchain_agent_real.py         — full functional suite (this file)
├── test_langchain_sandbox_integration.py — sandbox mechanics (all mocked)
└── test_sandbox_backend_integration.py  — AioSandboxBackend Docker tests
```

## See Also

- [Core Module](core.md) - LLM Factory and configuration
- [Extra Module](extra.md) - Non-pipeline tooling: agent graphs, anonymization, BAML
- [RAG & Workflow](rag.md) - RAG pipelines, retrievers, Prefect flows
- [Configuration Guide](../config/README.md) - Detailed configuration
- [MCP Servers](mcp-servers.md) - Model Context Protocol
- [Sandbox Support](sandbox_support.md) - Sandboxed execution
- [Deer-flow Integration](deer-flow.md) - ByteDance agent framework
- [Testing Guide](TESTING_GUIDE.md) - Testing patterns
- [Evaluation Testing](evaluation_testing.md) - LLM quality evaluation with openevals
