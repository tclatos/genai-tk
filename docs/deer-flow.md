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

# Chat with a profile (interactive REPL)
uv run cli agents run chat --chat

# Single-shot query
uv run cli agents run chat "tell me a joke"

# List available profiles (both harnesses)
uv run cli agents list

# Use a specific profile and LLM
uv run cli agents run research -m gpt_41mini@openai --chat
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
uv run cli agents list
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

DeerFlow profiles run through the unified `cli agents run` / `cli agents list`
commands — the same entry points used for LangChain profiles. The harness is
auto-resolved from the profile's `harness: deerflow` field, so there is no
separate `cli agents deerflow` command.

```bash
cli agents run <profile> [QUERY] [OPTIONS]

# Interactive REPL with a profile
cli agents run chat --chat

# Single-shot: answer a query and exit
cli agents run chat "What is the capital of France?"
echo "tell me a joke" | cli agents run chat   # query via stdin

# Specific profile
cli agents run research --chat

# Override LLM
cli agents run chat -m gpt_41mini@openai "Your question"

# Reasoning mode / sandbox overrides (DeerFlow)
cli agents run research --mode ultra "Compare RAG vs FAISS"
cli agents run research --sandbox docker "Run this code"

# List all profiles (both harnesses)
cli agents list
```

### Options (DeerFlow-relevant)

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--chat` | `-c` | false | Interactive REPL (multi-turn) |
| `--llm` | `-m` | profile default | Override LLM (genai-tk ID or tag) |
| `--mode` | | profile default | Reasoning mode: `flash` `thinking` `pro` `ultra` |
| `--sandbox` | `-b` | profile default | Sandbox: `local` `docker` |
| `--mcp` | | | Add extra MCP server (repeatable) |
| `--thread-id` | `-t` | new | Conversation thread ID |
| `--trace` | | false | Show graph node execution trace |
| `--json` | | false | Print raw NDJSON events |
| `--verbose` | `-v` | false | Enable DEBUG logging |

> DeerFlow native-web-UI config generation (`--generate-config`) is no longer a
> CLI flag. DeerFlow setup is covered by `cli init --extra harnessing`.

### Chat commands (in REPL)

| Command | Action |
|---------|--------|
| `/info` | Show current agent (harness, profile, model) |
| `/clear` | Start a new conversation thread |
| `/help` | Show help |
| `/quit` | Exit |

> Mid-session mode switching (`/mode`) and trace toggling (`/trace`) from the
> old DeerFlow-specific REPL are not in the unified REPL — pass `--mode` /
> `--trace` when starting `cli agents run ... --chat`.

### Modes

| Mode | Thinking | Planning | Sub-agents | Latency | Use for |
|------|----------|----------|-----------|---------|---------|
| `flash` | ✗ | ✗ | ✗ | fast | Quick Q&A, facts |
| `thinking` | ✓ | ✗ | ✗ | medium | Complex reasoning |
| `pro` | ✓ | ✓ | ✗ | slow | Research, analysis |
| `ultra` | ✓ | ✓ | ✓ | very slow | Multi-step research |

---

## Configuration

DeerFlow profiles live in the unified `agents:` dict (same file/dir as LangChain
profiles), each tagged `harness: deerflow`. DeerFlow **runtime settings** (skills
mount, `general` title/summarization/memory, `recursion_limit`, `default_profile`)
live separately in `config/deerflow.yaml` — see `config/examples/deerflow.yaml`.

### Minimal example

```yaml
# config/agents.yaml (or config/agents/deerflow.yaml)
agents:
  chat:
    harness: deerflow
    name: "chat"
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

  research:
    harness: deerflow
    name: "research"
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

# config/deerflow.yaml — runtime settings (not profiles)
deerflow:
  default_profile: "chat"
  skills:
    directories:
      - ${paths.project}/skills
```

### Full reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Display name; matched by `-p` (the dict key/slug is the canonical profile id) |
| `description` | string | | Short description shown in `--list` and the UI |
| `mode` | string | flash | Agent mode: `flash` `thinking` `pro` `ultra` |
| `llm` | string | | LLM ID (genai-tk format). Omit to use server default |
| `sandbox` | string | local | Sandbox type: `local` or `docker` |
| `subagent_enabled` | bool | false | Enable sub-agents (mode=ultra only) |
| `plan_mode` | bool | false | Enable planning (mode=pro/ultra) |
| `tool_groups` | list | `[bash]` | Enable tools: `web` `file:read` `file:write` `bash` |
| `mcp_servers` | list | `[]` | MCP server names from `config/mcp_servers.yaml` |
| `skill_directories` | list | | Paths to skill SKILL.md files (loaded recursively) |
| `available_skills` | list | all | Restrict skills by name (omit to allow all) |
| `middlewares` | list | `[]` | Same shape as LangChain agent profiles — `class` qualified name + kwargs. Reuses any LangChain `AgentMiddleware`, including the shared anonymization/routing middleware (see [middleware-pii-and-routing.md](middleware-pii-and-routing.md)) |
| `features` | list | | Display badges in UI (e.g., "🌐 Web Search") |
| `examples` | list | | Sample queries shown in UI |

---

## Architecture

```
┌─────────────────────────────────────────┐
│  genai-tk (this process)                │
│                                         │
│  cli agents run (harness.astream)       │
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

DeerFlow is also exposed through the shared harness layer
(`genai_tk.agents.harness`), which normalizes both DeerFlow and LangChain
behind one `BaseHarness` interface and one event model — use it via
`cli agents run <profile> "<query>"` or `create_harness(<profile>)` when you
don't want to special-case the runtime. See [agents.md](agents.md#harness-layer-agentsharness).

---

## Examples

### Example 1: Quick chat

```bash
uv run cli agents run chat --chat
```

Starts an interactive REPL. Type questions, use `/info`, `/clear`, etc.

### Example 2: Web research

```bash
uv run cli agents run research "Compare RAG vs FAISS for similarity search"
```

Single-shot query with web tools enabled (requires `tavily-mcp` + `TAVILY_API_KEY`).

### Example 3: Programming task

Create a profile:

```yaml
agents:
  coder:
    harness: deerflow
    mode: "thinking"
    tool_groups:
      - file:read
      - file:write
      - bash
```

Then:

```bash
uv run cli agents run coder "Debug and fix the import errors in my code"
```

### Example 4: Override LLM at runtime

```bash
uv run cli agents run research -m claude_haiku@openrouter --chat
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
agents:
  research:
    harness: deerflow
    mcp_servers:
      - tavily-mcp
```

Then:
```bash
export TAVILY_API_KEY=your_key
uv run cli agents run research --chat
```

**Q: The response is incomplete or truncated**

This can happen with very long outputs. Try switching to a different mode or LLM, or use `--sandbox docker` for more memory.

**Q: How do I filter which skills are available?**

Use `available_skills` in the profile:

```yaml
agents:
  limited:
    harness: deerflow
    available_skills:
      - public/web-search
      - custom/my-tool
```

**Q: Can I use DeerFlow with DeepAgents?**

Yes, but that requires a different integration. See the DeepAgents docs for `AioSandboxBackend` integration.
