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

# Generate native DeerFlow config files + start instructions for web UI
cli agents deerflow -p "Research Assistant" --generate-config

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
| `--generate-config` | `-G` | Generate config files + print launch instructions for native DeerFlow web UI |
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

### Quick path (recommended)

```bash
uv run cli init --deer-flow   # clones Deer-flow into ext/deer-flow and installs its deps
```

`DEER_FLOW_PATH` is set automatically to `<project>/ext/deer-flow` if not already in the environment.

Then run:

```bash
uv run cli agents deerflow -p "Research Assistant" --chat
```

### Manual path

```bash
git clone https://github.com/bytedance/deer-flow /path/to/deer-flow
cd /path/to/deer-flow/backend && uv sync
export DEER_FLOW_PATH=/path/to/deer-flow   # add to .env
```

Then configure a profile in `config/agents/deerflow.yaml` (see [Profiles](#profiles) below) and run as above.

## Architecture

```
┌────────────────────────────────────────────────────┐
│  genai-tk (this process)                           │
│                                                    │
│  cli_commands.py                                   │
│       │                                            │
│  EmbeddedDeerFlowClient  ←─ sys.path injection    │
│       │                    (DEER_FLOW_PATH/backend) │
│  src.client.DeerFlowClient (in-process)            │
│       ├── middlewares (injected from profile)      │
│       ├── available_skills (filtered from profile) │
│       └── SqliteSaver at data/kv_store/...        │
│                                                    │
│  config_bridge.py  ──► generates config.yaml      │
│                                                    │
│  [--generate-config]:                              │
│       ├─► config.yaml + extensions_config.json    │
│       └─► prints: langgraph dev / pnpm dev        │
└────────────────────────────────────────────────────┘
```

Before each run `config_bridge.setup_deer_flow_config()` generates:
- `<deer-flow>/backend/config.yaml` — model definitions from `config/providers/llm.yaml`
- `<deer-flow>/extensions_config.json` — MCP server wiring from `config/mcp_servers.yaml`

## Profiles

Profiles live in `config/agents/deerflow.yaml`.

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
    # Skill filtering: only expose these skills to the agent (omit for all)
    available_skills:
      - public/deep-research
      - public/data-analysis
    # Custom middlewares (Python qualified class name, no-arg constructor)
    middlewares:
      - mypackage.middleware.LoggingMiddleware

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
| `available_skills` | `null` | Restricts agent to these skill names only (null = all) |
| `middlewares` | `[]` | Python qualified class names to inject as agent middlewares |

## Middlewares

Middlewares allow injecting custom logic into the DeerFlow agent pipeline
(e.g. logging, guardrails, token tracking, audit trails).

### Writing a middleware

Middlewares must implement DeerFlow's `AgentMiddleware` interface:

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    """Records each agent invocation for auditing."""

    def on_agent_action(self, action, **kwargs):
        print(f"[AUDIT] Action: {action.tool} args={action.tool_input}")
        return action

    def on_agent_finish(self, finish, **kwargs):
        print(f"[AUDIT] Done: {finish.return_values}")
        return finish
```

Place this in a Python module on `sys.path`, then reference it in the profile:

```yaml
middlewares:
  - mypackage.middleware.LoggingMiddleware
```

The class is instantiated with no arguments at startup. Multiple middlewares
are stacked in list order.

### Skill filtering

`available_skills` limits which skills the agent may use:

```yaml
available_skills:
  - public/deep-research
  - custom/my-skill
```

When omitted (or `null`), all discovered skills are available.

## Native DeerFlow Web UI

Use `--generate-config` to export the DeerFlow-native config files then launch
the standard upstream DeerFlow stack (LangGraph backend + Next.js frontend):

```bash
# Generate config.yaml + extensions_config.json
cli agents deerflow -p "Research Assistant" --generate-config

# Terminal 1 — backend
cd $DEER_FLOW_PATH/backend
langgraph dev

# Terminal 2 — frontend
cd $DEER_FLOW_PATH/frontend
pnpm dev
```

Then open `http://localhost:3000`.

## Skills

Skills are Deer-flow "task templates" that extend what the agent can do (chart
generation, PPT creation, deep research, etc.). They are SKILL.md files loaded
at startup.

### Directory structure

```
skills/
├── public/       # Symlinks to deer-flow public skills
│   ├── chart-visualization -> $DEER_FLOW_PATH/skills/public/chart-visualization
│   └── ...
└── custom/       # Your own skills
    └── my-skill/
        └── SKILL.md
```

Link the public skills from your Deer-flow clone:
```bash
ln -s $DEER_FLOW_PATH/skills/public skills/public
```

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

Reference in profile:
```yaml
available_skills:
  - custom/my-skill
  - public/deep-research
```

## Embedded Client API

```python
from genai_tk.agents.deer_flow import EmbeddedDeerFlowClient
from genai_tk.utils.import_utils import instantiate_from_qualified_names

middlewares = instantiate_from_qualified_names(["mypackage.middleware.LoggingMiddleware"])
client = EmbeddedDeerFlowClient(
    config_path="/path/to/config.yaml",
    middlewares=middlewares,
    available_skills={"public/deep-research", "custom/my-skill"},
)
thread_id = "session-1"

async for event in client.stream_message(thread_id, "Explain transformer attention"):
    if isinstance(event, TokenEvent):
        print(event.data, end="", flush=True)
```

Access the upstream `DeerFlowClient` directly for memory, skills management, and MCP:

```python
# Direct upstream API access
client.client.get_memory_status()
client.client.list_skills()
client.client.update_skill("deep-research", enabled=True)
client.client.update_mcp_config({"tavily": {...}})
```

Multi-turn conversations reuse the same `thread_id`:
```python
await client.stream_message("user-123", "What is RAG?")
await client.stream_message("user-123", "How does it compare to fine-tuning?")
```

### Stream events

| Event | When emitted |
|-------|-------------|
| `TokenEvent(data)` | New text from the AI |
| `NodeEvent(node)` | Graph node activated (Planner, Researcher, Coder, Reporter) |
| `ToolCallEvent(tool_name, args, call_id)` | Agent called a tool |
| `ToolResultEvent(tool_name, content, call_id)` | Tool returned a result |
| `ClarificationEvent(question)` | Agent paused to ask a clarifying question |
| `ErrorEvent(message)` | An error occurred |

## Troubleshooting

**ImportError: cannot import DeerFlowClient**
- Verify `DEER_FLOW_PATH` is set and `$DEER_FLOW_PATH/backend/src/client.py` exists.
- Run `uv sync` in `$DEER_FLOW_PATH/backend`.

**ModuleNotFoundError: No module named 'readabilipy'**
- `uv add readabilipy` (required by Deer-flow's web_fetch tool).

**ImportError on middleware class**
- Ensure the middleware module is on `sys.path` / installed.
- Class must be a fully-qualified name: `module.ClassName`.

**Multi-turn memory lost after restart**
- The SqliteSaver checkpointer is in-process; restarting genai-tk starts fresh.
- Use `--chat` and keep the process running for long conversations.
