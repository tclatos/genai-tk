# Deer-flow Integration

[Deer-flow](https://github.com/bytedance/deer-flow) (ByteDance) is a LangGraph-based
multi-agent system with planning, sub-agents, and web research capabilities.
genai-tk integrates it as an **in-process embedded client** — no separate server needed
for terminal usage.

## Quick Reference

```bash
# Interactive chat (recommended)
cli agents deerflow --chat

# Single-shot query
cli agents deerflow "What is the weather in Toulouse?"

# Specific profile
cli agents deerflow -p "Research Assistant" --chat

# Override LLM at runtime
cli agents deerflow -p "Coder" -m gpt_41@openai --chat

# Start with web UI (Next.js + LangGraph backend)
cli agents deerflow -p "Research Assistant" --web

# List configured profiles
cli agents deerflow --list

# Read from stdin
echo "Summarise this text: ..." | cli agents deerflow
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--profile NAME` | `-p` | Profile from `deerflow.yaml` |
| `--chat` | | Interactive multi-turn REPL |
| `--llm ID` | `-m` | LLM override (genai-tk ID or tag) |
| `--mcp NAME` | | Extra MCP server (repeatable) |
| `--mode MODE` | | Override mode: `flash` `thinking` `pro` `ultra` |
| `--trace` | | Show graph node names as the agent works |
| `--list` | | Print profile table and exit |
| `--web` | | Start Deer-flow backend + Next.js web UI (localhost:3000) |
| `--verbose` | `-v` | Enable DEBUG logging |

### Chat Commands

| Command | Action |
|---------|--------|
| `/info` | Show current config (profile, mode, LLM, thread ID) |
| `/clear` | Start a new conversation thread |
| `/trace` | Toggle node-level trace on/off |
| `/quit` | Exit |

### Modes

| Mode | Thinking | Planning | Sub-agents | Use for |
|------|----------|----------|-----------|---------|
| `flash` | ✗ | ✗ | ✗ | Quick factual questions |
| `thinking` | ✓ | ✗ | ✗ | Reasoning-heavy tasks |
| `pro` | ✓ | ✓ | ✗ | Research, analysis |
| `ultra` | ✓ | ✓ | ✓ | Complex multi-step research |

## Setup

### 1. Clone Deer-flow

```bash
git clone https://github.com/bytedance/deer-flow /path/to/deer-flow
export DEER_FLOW_PATH=/path/to/deer-flow
```

Add `DEER_FLOW_PATH` to your `.env` or shell profile.

### 2. Install Deer-flow dependencies

```bash
cd $DEER_FLOW_PATH/backend
uv sync    # or: pip install -r requirements.txt
```

### 3. Configure a profile

Edit `config/basic/agents/deerflow.yaml` (see [Profiles](#profiles) below).

### 4. Run

```bash
cli agents deerflow -p "Research Assistant" --chat
```

No external servers are required for terminal mode.

## Architecture

```
┌────────────────────────────────────────────────────┐
│  genai-tk (this process)                           │
│                                                    │
│  cli_commands.py                                   │
│       │                                            │
│  EmbeddedDeerFlowClient  ←─ sys.path injection    │
│       │                    (DEER_FLOW_PATH/backend) │
│  DeerFlowClient (in-process, no HTTP)             │
│       │                                            │
│  config_bridge.py  ──► generates config.yaml      │
│       └── SqliteSaver at data/kv_store/...        │
│                                                    │
│  [--web only]:                                     │
│       ├─► LangGraph Server :2024                  │
│       └─► Gateway API :8001                       │
│           ↑ Next.js frontend @ localhost:3000      │
└────────────────────────────────────────────────────┘
```

Before each run `config_bridge.setup_deer_flow_config()` generates:
- `<deer-flow>/backend/config.yaml` — model definitions from `config/providers/llm.yaml`
- `<deer-flow>/extensions_config.json` — MCP server wiring from `config/mcp_servers.yaml`

## Profiles

Profiles live in `config/basic/agents/deerflow.yaml`.

```yaml
deerflow_agents:
  - name: Research Assistant
    description: Deep research with web access
    mode: pro
    llm: gpt_oss120@openrouter      # optional; uses server default if omitted
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
  skills:
    directories:
      - ${paths.project}/skills
    container_path: /mnt/skills
    trace_loading: true
```

### Profile fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | required | Display name |
| `mode` | `flash` | Agent mode |
| `llm` | `null` | LLM override (genai-tk ID or tag) |
| `mcp_servers` | `[]` | MCP server names from `mcp_servers.yaml` |
| `skill_directories` | `[]` | Directories to auto-discover skills from |
| `auto_start` | `true` | Auto-start servers when using `--web` |

## Skills

Skills are Deer-flow "task templates" that extend what the agent can do (chart
generation, PPT creation, deep research, etc.). They are SKILL.md files loaded
by the Gateway server.

### Directory structure

```
skills/
├── public/       # Symlinks to deer-flow public skills
│   ├── chart-visualization -> $DEER_FLOW_PATH/skills/public/chart-visualization
│   ├── data-analysis
│   └── ... (15 total)
└── custom/       # Your own skills
    └── README.md
```

Link the public skills from your Deer-flow clone:
```bash
ln -s $DEER_FLOW_PATH/skills/public skills/public
```

### Available public skills

`chart-visualization`, `consulting-analysis`, `data-analysis`, `deep-research`,
`find-skills`, `frontend-design`, `github-deep-research`, `image-generation`,
`podcast-generation`, `ppt-generation`, `skill-creator`, `surprise-me`,
`vercel-deploy-claimable`, `video-generation`, `web-design-guidelines`.

### Creating a custom skill

```bash
mkdir -p skills/custom/my-skill
```

Create `skills/custom/my-skill/SKILL.md`:
```markdown
---
name: my-skill
description: Brief description for the LLM
---

# My Skill

Instructions for the agent...
```

Enable in profile:
```yaml
skills:
  - custom/my-skill   # custom/ prefix for custom skills
  - deep-research     # public skills need no prefix
```

Skills not found on the server are silently skipped (logged at DEBUG level).

## Embedded Client API

```python
from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

client = EmbeddedDeerFlowClient(config_path="/path/to/config.yaml")
thread_id = "session-1"

async for event in client.stream_message(thread_id, "Explain transformer attention"):
    if isinstance(event, TokenEvent):
        print(event.data, end="", flush=True)
```

Multi-turn conversations reuse the same `thread_id`:
```python
await client.stream_message("user-123", "What is RAG?")
await client.stream_message("user-123", "How does it compare to fine-tuning?")
```

### Stream events

| Event | When emitted |
|-------|-------------|
| `TokenEvent(data)` | New text token from the AI |
| `NodeEvent(node)` | Graph node activated (Planner, Researcher, Coder, Reporter) |
| `ToolCallEvent(tool_name, args, call_id)` | Agent called a tool |
| `ToolResultEvent(tool_name, content, call_id)` | Tool returned a result |
| `ErrorEvent(message)` | An error occurred |

## Troubleshooting

**ImportError: cannot import DeerFlowClient**
- Verify `DEER_FLOW_PATH` is set and `$DEER_FLOW_PATH/backend/src/client.py` exists.

**ModuleNotFoundError: No module named 'readabilipy'**
- `uv add readabilipy` (required by Deer-flow's web_fetch tool).

**File output not accessible (Docker sandbox)**
- Files land inside the container at `/mnt/user-data/outputs/`.
- Use `--web` flag which mounts the outputs directory to the host.

**Stale model in web UI**
- Deer-flow saves the last-used model in `localStorage`.
- If the backend model list changes, the stale name causes `ValueError: Model not found`.
- Fix: clear browser localStorage for `localhost:3000`, or see
  [design/deer_flow_input_box_patch.md](design/deer_flow_input_box_patch.md) for the frontend patch.

**Multi-turn memory lost after restart**
- The SqliteSaver checkpointer is in-process; restarting genai-tk starts fresh.
- Use `--chat` and keep the process running for long conversations.
