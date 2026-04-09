# Agents Module (`genai_tk.agents`)

## Overview

The `agents` module provides agent implementations including LangChain-based agents (ReAct, Deep, Custom) and SmolAgents. All agents are configuration-driven with support for tools, MCP servers, middlewares, and checkpointing.

## LangChain Agents (`agents.langchain`)

### Unified Architecture

The LangChain agent module uses a **unified configuration system** supporting three agent types:

- **ReAct** - Standard reasoning agent with tool use loops
- **Deep** - Advanced reasoning with planning and subagents (requires `deepagents` package)
- **Custom** - Functional API-based custom agent from scratch

All agent types are managed through a single configuration interface with sensible defaults and profile-based customization.

### Configuration System

**Config File:** `config/langchain.yaml`

```yaml
langchain_agents:
  # Global defaults applied to all profiles
  defaults:
    type: react                          # default: react | deep | custom
    llm: null                            # null = use default from config
    enable_planning: true                # for deep agents only
    enable_file_system: true             # for deep agents only
    tools: []                            # default tools
    middlewares: []                      # middleware pipeline
    checkpointer:
      type: none                         # none | memory | postgres
    backend:
      type: none                         # none | aio_sandbox | class
    mcp_servers: []                      # model context protocol servers
    skills:
      directories: []                    # skill search directories

  # Default profile to use when none specified
  default_profile: "Research"

  # Named profiles for different use cases
  profiles:
    - name: "Research"
      type: deep
      llm: gpt_41@openai
      enable_planning: true
      tools:
        - spec: web_search
          config:
            provider: serper
      mcp_servers: []
      
    - name: "Coding"
      type: react
      llm: gpt_4o@openai
      enable_file_system: true
      tools:
        - spec: filesystem_tools
      middlewares:
        - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
          details: true
      
    - name: "DataAnalysis"
      type: react
      llm: gpt_4o@openai
      tools:
        - spec: sql_tools
          config:
            database: analytics.db
        - spec: dataframe_tools
```

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

**Configuration:**
```yaml
profiles:
  - name: "General"
    type: react
    llm: gpt_4o@openai
    tools:
      - spec: web_search
      - spec: calculator
```

**Usage:**
```python
from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
from genai_tk.agents.langchain.factory import create_langchain_agent

config = load_unified_config()
profile = resolve_profile(config, "General")
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

**Configuration:**
```yaml
profiles:
  - name: "Research"
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

**Note:** Deep agents require the `deepagents` library. Install with extra dependencies.

#### Custom Agent

Functional API-based agent built with LangGraph's Functional API for maximum customization.

**Best for:** Specialized workflows, advanced graph topologies

**Configuration:**
```yaml
profiles:
  - name: "Custom"
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

**Interactive Shell:**
```bash
# Use default profile with interactive chat
cli agents langchain --chat

# Use specific profile
cli agents langchain -p Research --chat

# Single query with specific profile
cli agents langchain -p Coding "List Python files"
```

**Single-Shot Queries:**
```bash
# Default profile, single query
cli agents langchain "What is machine learning?"

# Override agent type
cli agents langchain -p General --type react "Tell me a joke"

# Override LLM
cli agents langchain -p Research --llm gpt_4o@openai "Research AI"

# List available profiles
cli agents langchain --list
```

**Shell Mode:**
```python
from genai_tk.agents.langchain.agent_cli import run_langchain_agent_shell

# Start interactive agent shell
await run_langchain_agent_shell()
```

## SmolAgents (`agents.smolagents`)

CodeAct-based agent framework with code execution and tool composition.

**Key Features:**
- Code-based agents (generates Python code)
- Integrated tool execution
- Sandbox support (local, Docker, e2b)
- Model agnostic (works with any LLM)
- Interactive and batch modes

**CLI Usage:**
```bash
# Interactive shell mode
cli agents smol --chat

# Single query with tools
cli agents smol "How many users in the database?" -t sql_tools

# Web search and calculations
cli agents smol "What's the latest AI news and calculate 2+2?" -t web_search

# With custom LLM
cli agents smol "Your task" --llm gpt_4o@openai -t web_search

# With sandbox execution
cli agents smol "Run some Python code" --sandbox docker
```

**Tool Options:**
- `web_search` - Web search tool
- `calculator` - Math operations
- `sql_tools` - Database operations
- `dataframe_tools` - Data processing
- `yfinance_tools` - Financial data
- `browser_use` - Browser automation

**Configuration:**
SmolAgents configuration is in `config/smolagents.yaml`:
```yaml
codeact_agent:
  default:
    type: codeact
    llm: gpt_4o@openai
    tools:
      - web_search
      - calculator
```

## Common Patterns

### Pattern 1: Profile-Based Agent Selection

```python
from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
from genai_tk.agents.langchain.factory import create_langchain_agent

# Load configuration
config = load_unified_config()

# Select profile (e.g., from CLI, environment, or hardcoded)
profile_name = "Research"  # or get from args
profile = resolve_profile(config, profile_name)

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

Agent profiles live in `config/agents/langchain.yaml`.
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
cli agents langchain -p Research "Your query" --trace
```

**Inspect Configuration:**
```python
from genai_tk.agents.langchain.config import load_unified_config
config = load_unified_config()
print(config.model_dump_json(indent=2))
```

## Testing LangchainAgent

The integration test suite is in
`tests/integration_tests/agents/test_langchain_agent_real.py` and covers four
areas:

| Area | What is tested | Requires |
|---|---|---|
| React agents | Code generation, Q&A, streaming, multi-turn memory | `--include-real-models` |
| Deep agents (local) | Code generation, file writes via `FilesystemBackend` | `--include-real-models` |
| Named profiles | Field types, `enable_*` flags, profile resolution | (structural — no LLM) |
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
- [Extra Module](extra.md) - Advanced graphs and RAG
- [Configuration Guide](../config/README.md) - Detailed configuration
- [MCP Servers](mcp-servers.md) - Model Context Protocol
- [Sandbox Support](sandbox_support.md) - Sandboxed execution
- [Deep Agents CLI](deepagents-cli_integration.md) - Deepagents integration
- [Deer-flow Integration](deer-flow.md) - ByteDance agent framework
- [Testing Guide](TESTING_GUIDE.md) - Testing patterns
- [Evaluation Testing](evaluation_testing.md) - LLM quality evaluation with openevals
