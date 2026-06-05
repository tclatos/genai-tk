# Deer-flow Integration

[Deer-flow](https://github.com/bytedance/deer-flow) (ByteDance) is a LangGraph-based
multi-agent system with reasoning, planning, sub-agents, and web research.
genai-tk integrates it as an **in-process embedded client** via the `deerflow-harness` package.

---

## Quick start

```bash
# Install deerflow-harness into your project
cli init --with-deer-flow
# or:
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"

# Chat with the default profile (chat or research)
uv run cli agents deerflow --chat

# Single-shot query
uv run cli agents deerflow -i "tell me a joke"

# List available profiles
uv run cli agents deerflow --list

# Use a specific profile and LLM
uv run cli agents deerflow -p research -m gpt_41mini@openai --chat
```

---

## Features

- **Embedded execution** — no separate server needed for terminal/script usage
- **In-process async** — all communication in-memory via async events
- **Multi-agent planning** — reasoning and planning modes (flash, thinking, pro, ultra)
- **Web research** — tavily-mcp integration for live web search
- **Skills** — domain-specific SKILL.md files loaded on demand
- **MCP servers** — extensible via Model Context Protocol
- **Thread persistence** — multi-turn conversations saved via SqliteSaver

---

## Installation

### With `cli init` (recommended)

```bash
uv run cli init --with-deer-flow

# This installs deerflow-harness via uv
# Then you can run:
uv run cli agents deerflow --list
```

### Manual

```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
```

To update to the latest version:

```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness" --force
```

Or pin to a specific commit:

```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@abc1234#subdirectory=backend/packages/harness"
```

---

## CLI reference

### Main commands

```bash
cli agents deerflow [OPTIONS] [QUERY]

# No arguments → interactive chat with default profile
cli agents deerflow --chat

# Single-shot: answer a query and exit
cli agents deerflow "What is the capital of France?"
cli agents deerflow -i "tell me a joke"   # same as above

# Specific profile
cli agents deerflow -p research --chat

# Override LLM
cli agents deerflow -m gpt_41mini@openai "Your question"

# List all profiles
cli agents deerflow --list
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--profile` | `-p` | default from config | Profile name from `deerflow.yaml` |
| `--chat` | | false | Interactive REPL (multi-turn) |
| `--llm` | `-m` | profile default | Override LLM (genai-tk ID or tag) |
| `--mcp` | | | Add extra MCP server (repeatable) |
| `--mode` | | profile default | Override reasoning mode: `flash` `thinking` `pro` `ultra` |
| `--sandbox` | | profile default | Override sandbox: `local` `docker` |
| `--trace` | | false | Show graph node execution trace |
| `--list` | | | Print profiles and exit |
| `--verbose` | `-v` | false | Enable DEBUG logging |
| `--generate-config` | | | Generate config files for native DeerFlow web UI |

### Chat commands (in REPL)

| Command | Action |
|---------|--------|
| `/info` | Show current config (profile, mode, LLM, thread ID, models) |
| `/mode <flash\|thinking\|pro\|ultra>` | Switch reasoning mode (no restart) |
| `/trace` | Toggle node-level trace |
| `/clear` | Start a new conversation thread |
| `/help` | Show help |
| `/quit` | Exit |

### Modes

| Mode | Thinking | Planning | Sub-agents | Latency | Use for |
|------|----------|----------|-----------|---------|---------|
| `flash` | ✗ | ✗ | ✗ | fast | Quick Q&A, facts |
| `thinking` | ✓ | ✗ | ✗ | medium | Complex reasoning |
| `pro` | ✓ | ✓ | ✗ | slow | Research, analysis |
| `ultra` | ✓ | ✓ | ✓ | very slow | Multi-step research |

---

## Configuration

Profiles live in `config/agents/deerflow.yaml`.

### Minimal example

```yaml
deerflow_agents:
  - name: "chat"
    description: "Lightweight chat (no tools)"
    mode: "flash"
    sandbox: local
    tool_groups:
      - bash
    mcp_servers: []
    features:
      - "⚡ Fast Mode"
    examples:
      - "Tell me a joke"
      - "Write a Python function"

  - name: "research"
    description: "Web research with planning"
    mode: "pro"
    sandbox: local
    tool_groups:
      - web
      - bash
    mcp_servers:
      - tavily-mcp      # requires TAVILY_API_KEY
    features:
      - "🌐 Web Search"
      - "🧠 Planning"
    examples:
      - "Research the latest AI developments"

deerflow:
  default_profile: "chat"
  skills:
    directories:
      - ${paths.project}/skills
```

### Full reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Profile key (used with `-p research`) |
| `description` | string | required | Display name in `--list` |
| `mode` | string | flash | Agent mode: `flash` `thinking` `pro` `ultra` |
| `llm` | string | | LLM ID (genai-tk format). Omit to use server default |
| `sandbox` | string | local | Sandbox type: `local` or `docker` |
| `subagent_enabled` | bool | false | Enable sub-agents (mode=ultra only) |
| `plan_mode` | bool | false | Enable planning (mode=pro/ultra) |
| `tool_groups` | list | `[bash]` | Enable tools: `web` `file:read` `file:write` `bash` |
| `mcp_servers` | list | `[]` | MCP server names from `config/mcp_servers.yaml` |
| `skill_directories` | list | | Paths to skill SKILL.md files (loaded recursively) |
| `available_skills` | list | all | Restrict skills by name (omit to allow all) |
| `middlewares` | list | `[]` | Python qualified class names (no-arg constructors) |
| `features` | list | | Display badges in UI (e.g., "🌐 Web Search") |
| `examples` | list | | Sample queries shown in UI |

---

## Architecture

```
┌─────────────────────────────────────────┐
│  genai-tk (this process)                │
│                                         │
│  cli_commands._run_single_shot()        │
│       │                                 │
│  EmbeddedDeerFlowClient                 │
│       │  config_path, model_name        │
│       │                                 │
│  deerflow.client.DeerFlowClient         │
│       │  (in-process)                   │
│       ├─ middlewares (injected)         │
│       ├─ available_skills (filtered)    │
│       └─ SqliteSaver checkpointer       │
│                                         │
│  config_bridge.setup_deer_flow_config() │
│       └─► generates config.yaml         │
│           (model list, sandbox, etc.)   │
└─────────────────────────────────────────┘
```

No `DEER_FLOW_PATH` env var needed. The `deerflow-harness` package is installed
like any other Python dependency via `uv add`.

---

## Examples

### Example 1: Quick chat

```bash
uv run cli agents deerflow -p chat --chat
```

Starts an interactive REPL. Type questions, use `/info`, `/clear`, etc.

### Example 2: Web research

```bash
uv run cli agents deerflow -p research "Compare RAG vs FAISS for similarity search"
```

Single-shot query with web tools enabled (requires `tavily-mcp` + `TAVILY_API_KEY`).

### Example 3: Programming task

Create a profile:

```yaml
deerflow_agents:
  - name: "coder"
    mode: "thinking"
    tool_groups:
      - file:read
      - file:write
      - bash
```

Then:

```bash
uv run cli agents deerflow -p coder "Debug and fix the import errors in my code"
```

### Example 4: Override LLM at runtime

```bash
uv run cli agents deerflow -p research -m claude_haiku@openrouter --chat
```

Uses the Claude Haiku model instead of the profile's default LLM.

---

## Troubleshooting

**Q: "deerflow-harness is not installed"**

Install it:
```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
```

**Q: How do I use web search?**

Add `tavily-mcp` to `mcp_servers:` and set the `TAVILY_API_KEY` env var:

```yaml
deerflow_agents:
  - name: research
    mcp_servers:
      - tavily-mcp
```

Then:
```bash
export TAVILY_API_KEY=your_key
uv run cli agents deerflow -p research --chat
```

**Q: The response is incomplete or truncated**

This can happen with very long outputs. Try switching to a different mode or LLM, or use `--sandbox docker` for more memory.

**Q: How do I filter which skills are available?**

Use `available_skills` in the profile:

```yaml
deerflow_agents:
  - name: limited
    available_skills:
      - public/web-search
      - custom/my-tool
```

**Q: Can I use DeerFlow with DeepAgents?**

Yes, but that requires a different integration. See the DeepAgents docs for `AioSandboxBackend` integration.
