# Deer-flow Integration Guide

## Overview

[Deer-flow](https://github.com/bytedance/deer-flow) (ByteDance) is a LangGraph-based multi-agent system with advanced reasoning, planning, and search capabilities.  The GenAI Toolkit integrates with it via an **HTTP client** that talks to a running Deer-flow server, keeping the two projects completely independent.

**Key design principles:**
- Deer-flow runs as a separate server process (LangGraph Server + Gateway API)
- GenAI Toolkit never imports deer-flow Python code at runtime
- Config bridge translates genai-tk YAML configs into deer-flow format before the server starts
- Profiles in `config/agents/deerflow.yaml` control mode, LLM, MCP servers and skills


## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  genai-tk  (this process)                               │
│                                                         │
│  CLI / Streamlit                                        │
│       │                                                 │
│  cli_commands.py  ──► config_bridge.py                  │
│       │               (writes deer-flow config.yaml     │
│       │                and extensions_config.json)      │
│       │                                                 │
│  DeerFlowClient  ──► HTTP ──► LangGraph Server :2024    │
│  (client.py)          │       (lead_agent graph)        │
│                       └─────► Gateway API       :8001   │
│                               (skills, MCP proxy)       │
└─────────────────────────────────────────────────────────┘
```

### Ports

| Service | Default port | Purpose |
|---------|-------------|---------|
| LangGraph Server | 2024 | Run/stream the `lead_agent` graph |
| Gateway API | 8001 | Skills management, MCP proxying |
| nginx (optional) | 2026 | Unified proxy for both services |


## Setup

### 1. Clone deer-flow

```bash
git clone https://github.com/bytedance/deer-flow /path/to/deer-flow
```

### 2. Set DEER_FLOW_PATH

```bash
export DEER_FLOW_PATH=/path/to/deer-flow
```

Add to your `.env` or shell profile. The server manager uses this to locate the backend directory and to start the servers.

### 3. Install deer-flow dependencies

```bash
cd /path/to/deer-flow/backend
uv sync          # or: pip install -r requirements.txt
```

### 4. Configure a profile

Edit `config/agents/deerflow.yaml` (see [Profiles](#profiles) below).

### 5. Run

```bash
cli agents deerflow --chat
# or single-shot:
cli agents deerflow "What is the weather in Toulouse?"

# Start the Deer-flow Next.js web UI
cli agents deerflow -p "Research Assistant" --web
# UI: http://localhost:3000
# Logs: $DEER_FLOW_PATH/logs/frontend.log
```

The servers are auto-started if not already running (requires `DEER_FLOW_PATH`).


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


## Config Bridge

Before the server starts, `config_bridge.setup_deer_flow_config()` generates:

- `<deer-flow>/backend/config.yaml` — model definitions from `config/providers/llm.yaml`
- `<deer-flow>/extensions_config.json` — MCP server wiring from `config/mcp_servers.yaml`

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


## HTTP Client

`DeerFlowClient` wraps the LangGraph HTTP API (`/threads`, `/runs/stream`) and the Gateway REST API (`/api/skills`).

```python
from genai_tk.extra.agents.deer_flow.client import DeerFlowClient, TokenEvent

client = DeerFlowClient()
thread_id = await client.create_thread()

async for event in client.stream_run(thread_id, "Explain transformer attention"):
    if isinstance(event, TokenEvent):
        print(event.data, end="", flush=True)
```

**Stream events:**

| Event class | When emitted |
|------------|-------------|
| `TokenEvent(data=str)` | New text from the AI message |
| `NodeEvent(node=str)` | A graph node became active (filtered to meaningful nodes) |
| `ErrorEvent(message=str)` | Server reported an error |

`stream_run` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thread_id` | required | Thread from `create_thread()` |
| `user_input` | required | User message |
| `model_name` | `None` | Deer-flow model name override |
| `thinking_enabled` | `False` | Enable chain-of-thought |
| `is_plan_mode` | `False` | Enable planning step |

> **LangGraph 0.6+ notes:**
> - `assistant_id: "lead_agent"` is required in every run body
> - Thread context is sent as a top-level `"context"` dict (not inside `config.configurable`)
> - SSE uses `event: messages/partial` with *accumulated* content (not incremental deltas);
>   the client tracks per-message content length and emits only new characters


## Server Manager

`DeerFlowServerManager` starts/stops the LangGraph and Gateway servers as subprocesses:

```python
from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

async with DeerFlowServerManager(deer_flow_path="/path/to/deer-flow") as mgr:
    # both servers are up here
    ...
```

- If both servers are already reachable, `start()` is a no-op
- Uses `httpx.AsyncHTTPTransport()` for health checks to bypass corporate HTTP proxies
- Injects `NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0` into subprocess environment
- Startup timeout: 60 s (first `langgraph dev` run may take longer due to package download)


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

### Server won't start

- Check `DEER_FLOW_PATH` points to the deer-flow clone root
- Verify the backend venv exists: `ls $DEER_FLOW_PATH/backend/.venv`
- Look at stdout/stderr from the server subprocess (enable `--verbose`)

### PowerPoint / package-install tasks fail with `PermissionError` on `.deer-flow/threads`

If agent tool calls fail with errors like:
- `PermissionError: [Errno 13] Permission denied: .../backend/.deer-flow/threads`

the issue is usually filesystem ownership of Deer-flow runtime state, not the package itself.

**Rationale:** sandbox tools (`bash`, `ls`, `read_file`, etc.) use per-thread directories under:
- `$DEER_FLOW_PATH/backend/.deer-flow/threads/{thread_id}/user-data/...`

If `.deer-flow` was created by `root` (for example after running setup/start with `sudo`), normal user runs cannot create thread working directories, and many unrelated tasks (including PPT generation) fail.

Fix permissions:

```bash
sudo chown -R $USER:$USER $DEER_FLOW_PATH/backend/.deer-flow
mkdir -p $DEER_FLOW_PATH/backend/.deer-flow/threads
chmod -R u+rwX $DEER_FLOW_PATH/backend/.deer-flow
```

Then restart Deer-flow.

### `pip install` in sandbox fails on Ubuntu (`externally-managed-environment`)

On Ubuntu, system Python follows PEP 668 and blocks direct installs into the OS-managed environment.

Typical error:
- `error: externally-managed-environment`

**Rationale:** project environments created by `uv` can have the package installed, while agent `bash` tool commands may still run against system Python unless a venv is explicitly used in-command.

Preferred approach inside agent/sandbox commands:

```bash
python3 -m venv /mnt/user-data/workspace/.venv
/mnt/user-data/workspace/.venv/bin/python -m ensurepip --upgrade
/mnt/user-data/workspace/.venv/bin/python -m pip install python-pptx
```

Then run Python scripts with `/mnt/user-data/workspace/.venv/bin/python ...`.

### PPT tasks looping into `GraphRecursionError`

If the model repeatedly retries install/debug commands, the run may hit:
- `GraphRecursionError: Recursion limit of 25 reached ...`

This is usually a downstream symptom of one of the two issues above (directory permissions or system-Python pip restrictions). Fix those first, then rerun the task.

### Corporate proxy timeouts

The client explicitly uses `httpx.AsyncHTTPTransport()` and the subprocess has `NO_PROXY` set, so localhost traffic never goes through a proxy.  If you still see timeouts, confirm `HTTP_PROXY` is not overriding per-request settings.

### Empty response / no tokens

If you see an empty assistant panel, ensure the deer-flow server version uses LangGraph ≥ 0.6 (check `$DEER_FLOW_PATH/backend/pyproject.toml`).  Older versions use `event: messages` (delta); 0.6+ uses `event: messages/partial` (accumulated) — both are handled by the client.

### HTTP 422 errors

Ensure `assistant_id: "lead_agent"` is accepted by the server.  If the graph was renamed, update `DeerFlowClient.__init__` parameter `assistant_id`.
