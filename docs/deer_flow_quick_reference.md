# Deer-Flow Agent — Quick Reference

## Usage

```bash
# Interactive chat (recommended)
cli agents deerflow --chat

# Single-shot query
cli agents deerflow "What is the weather in Toulouse?"

# Specific profile
cli agents deerflow -p "Research Assistant" --chat

# Override LLM
cli agents deerflow -p "Coder" --llm gpt_41@openai --chat

# Override mode
cli agents deerflow -p "Coder" --mode ultra "Refactor this function"

# Extra MCP servers on top of profile
cli agents deerflow --mcp github --chat

# Show node-level execution trace
cli agents deerflow --trace "Analyse AI trends"

# List configured profiles
cli agents deerflow --list

# Read from stdin
echo "Summarise this text: ..." | cli agents deerflow

# Verbose / debug logging
cli agents deerflow --verbose --chat

# Open native web client (starts Next.js frontend + backend, opens browser)
cli agents deerflow -p "Research Assistant" --web
```

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--profile NAME` | `-p` | Profile from `deerflow.yaml` (default from config or first) |
| `--chat` | `-s` | Interactive multi-turn REPL |
| `--llm ID` | `-m` | LLM override (genai-tk ID or tag) |
| `--mcp NAME` | | Extra MCP server (repeatable) |
| `--mode MODE` | | Override mode: `flash` `thinking` `pro` `ultra` |
| `--trace` | | Show graph node names as the agent works |
| `--list` | | Print profile table and exit |
| `--web` | | Start native Next.js web client and open browser |
| `--verbose` | `-v` | Enable DEBUG logging |


## Chat Commands

| Command | Action |
|---------|--------|
| `/help` | Show command list |
| `/info` | Show current config (profile, mode, LLM, thread ID) |
| `/clear` | Start a new conversation thread |
| `/trace` | Toggle node-level trace on/off |
| `/quit` `/exit` `/q` | Exit |

History is saved in `.deerflow.input.history` (navigate with ↑ ↓).


## Modes

| Mode | Capability | Use for |
|------|-----------|---------|
| `flash` | Direct answer | Quick lookup, simple questions |
| `thinking` | Chain-of-thought | Reasoning, math, logic |
| `pro` | Planning + thinking | Research, analysis, writing |
| `ultra` | Planning + sub-agents | Multi-step complex research |


## Startup Output

```
╭─── Deer-flow Interactive Chat ───╮
Profile: Research Assistant  Mode: pro  LLM: gpt_oss120@openrouter
MCP: tavily-mcp

Commands: /quit /exit /q  /clear  /help  /info  /trace
```

During processing, step labels appear above the live answer panel:

```
[11:32:45] → Planning
[11:32:48] → Researching
╭──────────────── Assistant ────────────────╮
│ The current weather in Toulouse is ...    │
╰───────────────────────────────────────────╯
```


## Profile Configuration

Edit `config/agents/deerflow.yaml`:

```yaml
deerflow_agents:
  - name: Research Assistant
    description: Deep research with web access
    mode: pro                           # flash | thinking | pro | ultra
    llm: gpt_oss120@openrouter          # optional; uses server default if omitted
    mcp_servers:
      - tavily-mcp
    skill_directories:
      - ${paths.project}/skills
    auto_start: true                    # auto-start servers if not running

  - name: Coder
    description: Code analysis and generation
    mode: flash

deerflow:
  default_profile: Research Assistant
```

> Server URLs (`langgraph_url`, `gateway_url`) default to `localhost:2024` / `localhost:8001`
> and rarely need changing in development.


## Server Auto-Start

When `auto_start: true` and the server is not running, the CLI:
1. Reads `DEER_FLOW_PATH` to find the deer-flow clone
2. Writes `config.yaml` + `extensions_config.json` into `$DEER_FLOW_PATH/backend/`
3. Launches `langgraph dev` and the Gateway API as background processes
4. Waits up to 60 s for both to become healthy

Set the environment variable:
```bash
export DEER_FLOW_PATH=/path/to/deer-flow   # add to .env or shell profile
```

To use a server you started manually, set `auto_start: false`.


## Related Docs

- [Deer_Flow_Integration.md](Deer_Flow_Integration.md) — architecture, setup, HTTP client API
- [deer_flow_skills_management.md](deer_flow_skills_management.md) — skills directory, custom skills
- [mcp-servers.md](mcp-servers.md) — MCP server configuration
