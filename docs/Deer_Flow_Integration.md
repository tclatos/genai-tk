# Deer-flow Integration Guide

## Overview

[Deer-flow](https://github.com/bytedance/deer-flow) (ByteDance) is a LangGraph-based multi-agent system with advanced reasoning, planning, and search capabilities. The GenAI Toolkit integrates with it via an **embedded in-process client** that loads Deer-flow directly from `DEER_FLOW_PATH/backend`.

**Key design principles:**
- Deer-flow Python code runs in-process (no separate server needed for terminal usage)
- Configuration is generated on-demand before each run via `config_bridge`
- Profiles in `config/agents/deerflow.yaml` control mode, LLM, MCP servers and skills
- The `--web` option starts the backend servers for the Next.js frontend at `http://localhost:3000`


## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  genai-tk (this process)                                     │
│                                                              │
│  CLI / Streamlit                                             │
│       │                                                      │
│  cli_commands.py                                             │
│       │                                                      │
│  EmbeddedDeerFlowClient      ← sys.path injection           │
│  (embedded_client.py)         (DEER_FLOW_PATH/backend)      │
│       │                                                      │
│  DeerFlowClient              ← imported from src.client      │
│  (in-process, no HTTP)                                       │
│       │                                                      │
│  config_bridge.py            ← generates config.yaml         │
│  (writes to DEER_FLOW_PATH/backend)                         │
│       └──► SqliteSaver (checkpointer at data/kv_store/...)  │
│           (multi-turn memory stays in genai-tk session)      │
│                                                              │
│  [--web flag only]: starts servers for frontend             │
│       │                                                      │
│       ├─► LangGraph Server :2024                            │
│       └─► Gateway API :8001                                 │
│           ↑ Next.js frontend @ http://localhost:3000        │
└──────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Clone deer-flow

```bash
git clone https://github.com/bytedance/deer-flow /path/to/deer-flow
```

### 2. Set DEER_FLOW_PATH

```bash
export DEER_FLOW_PATH=/path/to/deer-flow
```

Add to your `.env` or shell profile. The embedded client uses this to find the backend Python code.

### 3. Install deer-flow dependencies

```bash
cd /path/to/deer-flow/backend
uv sync          # or: pip install -r requirements.txt
```

### 4. Configure a profile

Edit `config/agents/deerflow.yaml` (see [Profiles](#profiles) below).

### 5. Run (terminal mode — no servers needed)

```bash
# Single-shot query
cli agents deerflow "What is the weather in Toulouse?"

# Interactive chat
cli agents deerflow -p "Research Assistant" --chat
```

The embedded client handles everything in-process. No servers are required.

### 6. (Optional) Run with the web UI

```bash
cli agents deerflow -p "Research Assistant" --web
# Opens http://localhost:3000
# Backend servers auto-start (if `auto_start: true`)
```



## Profiles

Profiles live in `config/agents/deerflow.yaml` and are loaded by `DeerFlowProfile` (a Pydantic model).

```yaml
deerflow_agents:
  - name: Research Assistant
    description: Deep research with web access
    mode: pro
    llm: gpt_oss120@openrouter      # optional override; uses server default if omitted
    mcp_servers:
      - tavily-mcp
    skill_directories:
      - ${paths.project}/skills
    auto_start: true

  - name: Coder
    description: Code analysis and generation
    mode: flash
    mcp_servers: []

deerflow:
  default_profile: Research Assistant
```

### Profile fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | required | Display name |
| `mode` | `flash` | Agent mode (see [Modes](#modes)) |
| `llm` | `null` | LLM override (genai-tk ID or tag) |
| `mcp_servers` | `[]` | MCP server names from `mcp_servers.yaml` |
| `skills` | `[]` | Explicit skill list (`category/name`) |
| `skill_directories` | `[]` | Directories to auto-discover skills from |
| `langgraph_url` | `http://localhost:2024` | LangGraph server URL |
| `gateway_url` | `http://localhost:8001` | Gateway API URL |
| `auto_start` | `true` | Auto-start servers if not running |
| `deer_flow_path` | `null` | Override for server start (falls back to `DEER_FLOW_PATH`) |


## Modes

| Mode | Thinking | Planning | Sub-agents | Use for |
|------|----------|----------|-----------|---------|
| `flash` | ✗ | ✗ | ✗ | Quick factual questions |
| `thinking` | ✓ | ✗ | ✗ | Reasoning-heavy tasks |
| `pro` | ✓ | ✓ | ✗ | Research, analysis |
| `ultra` | ✓ | ✓ | ✓ | Complex multi-step research |


## Config Bridge & Checkpointer

Before each run, `config_bridge.setup_deer_flow_config()` generates:

- `<deer-flow>/backend/config.yaml` — model definitions from `config/providers/llm.yaml`
- `<deer-flow>/extensions_config.json` — MCP server wiring from `config/mcp_servers.yaml`

The embedded client also creates a `SqliteSaver` checkpointer at `data/kv_store/deerflow_checkpoints.db` for persistent multi-turn conversation state.

**LLM translation example:**

```yaml
# genai-tk (llm.yaml)
openrouter:
  api_key_env: OPENROUTER_API_KEY
  models:
    - id: gpt_oss120@openrouter
      model: openai/gpt-4.1-mini
```

becomes in `config.yaml`:

```yaml
models:
  gpt_oss120@openrouter:
    model: openai/gpt-4.1-mini
    openai_api_base: https://openrouter.ai/api/v1
    openai_api_key: ${OPENROUTER_API_KEY}
```


## Embedded Client

`EmbeddedDeerFlowClient` runs Deer-flow in-process by:
1. Injecting `DEER_FLOW_PATH/backend` into Python's `sys.path`
2. Importing `DeerFlowClient` from `src.client` directly
3. Creating a `SqliteSaver` checkpointer at `data/kv_store/deerflow_checkpoints.db` for multi-turn memory
4. Translating Deer-flow `StreamEvent` objects into typed genai-tk events

```python
from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

client = EmbeddedDeerFlowClient(config_path="/path/to/config.yaml")
thread_id = "session-1"

async for event in client.stream_message(thread_id, "Explain transformer attention"):
    if isinstance(event, TokenEvent):
        print(event.data, end="", flush=True)
```

**Stream events:**

| Event class | When emitted |
|------------|-------------|
| `TokenEvent(data=str)` | New text from the AI message |
| `NodeEvent(node=str)` | A graph node became active (Planner, Researcher, Coder, Reporter) |
| `ToolCallEvent(tool_name, args, call_id)` | The agent called a tool |
| `ToolResultEvent(tool_name, content, call_id)` | A tool returned a result |
| `ErrorEvent(message=str)` | An error occurred |

`stream_message()` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thread_id` | required | Conversation thread ID (scopes checkpointer memory) |
| `user_input` | required | User message |
| `model_name` | `None` | Deer-flow model name override |
| `mode` | `"flash"` | Agent mode (`flash`, `thinking`, `pro`, `ultra`) |
| `subagent_enabled` | `None` | Enable sub-agent delegation (overrides mode default) |
| `plan_mode` | `None` | Enable planning mode (overrides mode default) |

**Multi-turn conversations:**

Create a new `thread_id` per session, reuse it for subsequent turns:

```python
thread_id = "user-123"

# Turn 1
await client.stream_message(thread_id, "What is RAG?")

# Turn 2 — the agent remembers context from turn 1
await client.stream_message(thread_id, "How does it compare to fine-tuning?")
```

Checkpointer state persists for the lifetime of the process. Restarting genai-tk loses the session memory.



## Server Manager (for --web only)

When using `--web` to launch the Next.js frontend, `DeerFlowServerManager` auto-starts the LangGraph and Gateway servers as subprocesses if not already running:

```python
from genai_tk.agents.deer_flow.server_manager import DeerFlowServerManager

mgr = DeerFlowServerManager(deer_flow_path="/path/to/deer-flow")
await mgr.restart()  # stop everything, clean, start fresh

# Both services are now available:
# - LangGraph :2024
# - Gateway :8001
# - nginx (optional) :2026 (unified proxy)
```

- If both services are already reachable, `is_running()` returns `True` and startup is skipped
- Uses `httpx.AsyncHTTPTransport()` for health checks (bypasses corporate proxies)
- Injects `NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0` into subprocess environment
- Startup timeout: 90 seconds (first `langgraph dev` run may take longer due to package downloads)



## Skills

Skills are deer-flow "task templates" loaded by the Gateway.  They extend what the agent can do (chart generation, PPT creation, deep research, etc.).

See [deer_flow_skills_management.md](deer_flow_skills_management.md) for the full skills guide.

**Quick setup:**

```bash
# Link all public skills from the deer-flow clone
ln -s $DEER_FLOW_PATH/skills/public genai-tk/skills/public
```

In the profile, set `skill_directories` and the CLI will enable matching skills on every run:

```yaml
skill_directories:
  - ${paths.project}/skills
```

Skills not found on the server are silently skipped (logged at DEBUG level).


## Troubleshooting

### Embedded client won't start

**Error: `ImportError: cannot import name 'DeerFlowClient' from 'src.client'`**

- Ensure `DEER_FLOW_PATH` is set: `echo $DEER_FLOW_PATH`
- Verify backend directory exists: `ls $DEER_FLOW_PATH/backend`
- Check that `src/client.py` exists: `ls $DEER_FLOW_PATH/backend/src/client.py`

**Error: `ModuleNotFoundError: No module named 'readabilipy'`**

- Install missing dependency: `uv add readabilipy`
- This is required by Deer-flow's `src.community.jina_ai.tools` (web_fetch)

**Error: `Cannot operate on a closed database`**

- The sqlite checkpointer connection was garbage-collected
- Ensure genai-tk is using the latest embedded_client.py (uses direct `sqlite3.connect()`, not context manager)
- Update with: `git pull`

### Web mode (--web) server issues

For `cli agents deerflow --web`, the backend servers must be running.

**Error: `Deer-flow server is not running`**

- If `auto_start: false` in the profile, start servers manually:
  ```bash
  cd $DEER_FLOW_PATH/backend
  make dev
  ```
- Or set `auto_start: true` in the profile and try again

**Checking server status:**

```bash
# LangGraph Server
curl http://localhost:2024/info

# Gateway API
curl http://localhost:8001/api/models

# Logs
tail -f $DEER_FLOW_PATH/logs/langgraph.log
tail -f $DEER_FLOW_PATH/logs/gateway.log
```

### Checkpointer state issues

**Multi-turn memory lost after restart**

- SqliteSaver state is in-process; restarting genai-tk starts a fresh session
- To persist across restarts, implement a session loader (not yet supported)
- For now, use `--chat` and keep the process running for long conversations

**Large checkpoints slow down `stream_message`**

- LangGraph saves the full conversation state on each turn
- On very long conversations (100+ turns), consider starting a new thread periodic

### File output not accessible

**Problem: Agent generates files (e.g., PowerPoint, plots) but they don't appear in expected location**

This happens because of how sandbox modes handle file access:

**Docker sandbox** (`sandbox: docker`):
- Files are created inside the Docker container at `/mnt/user-data/outputs/`
- Without proper volume mounts, the host cannot access these files
- **Solution: Use `--web` flag** — this starts the actual Docker sandbox with proper volume mapping
  ```bash
  cli agents deerflow -p "Research Assistant" --web
  # Files now appear on host filesystem
  ```

**Local sandbox** (`sandbox: local`):
- Lightweight sandboxing without Docker
- File paths may be isolated from the host
- **Solution: Modify tool configuration** to write to `~/.genai-tk/outputs/` or any shared directory

**Recommended fix:**
1. For file generation tasks, use: `cli agents deerflow -p "Research Assistant" --web`
2. Or modify profiles to set `sandbox: local` for simpler setup (less isolated)
3. Or configure agent to output to a shared directory (modify tool code)

### Mode and flag issues

**Error: `Unknown mode: 'foo'`**

Valid modes: `flash`, `thinking`, `pro`, `ultra`

Compare to profile:

```bash
cli agents deerflow --list
```

**Subagent / plan-mode not taking effect**

- CLI flags `--subagent` and `--plan-mode` override profile settings
- Check profile defaults:
  ```bash
  grep -A 5 "name: Research Assistant" config/agents/deerflow.yaml
  ```

