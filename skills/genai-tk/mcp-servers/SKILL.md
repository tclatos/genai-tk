---
name: mcp-servers
description: Expose genai-tk tools and agents as MCP servers, generate standalone MCP scripts, and debug MCP configuration and tool adaptation.
---

# GenAI Toolkit MCP Servers

## Read First

- `docs/mcp-servers.md`
- `genai_tk/mcp/config.py`
- `genai_tk/mcp/server_builder.py`
- `genai_tk/mcp/tool_adapter.py`
- `genai_tk/mcp/script_generator.py`
- `config/examples/tk_servers.yaml`

## Concepts

| Concept | Implementation |
|---|---|
| Server definition | `config/examples/tk_servers.yaml` under `mcp_expose_servers` |
| External server registry | `config/mcp_servers.yaml` under `mcpServers` |
| Tool factory loading | `genai_tk/mcp/config.py`, `genai_tk/mcp/tool_adapter.py` |
| Runtime server | `genai_tk/mcp/server_builder.py` |
| Agent-as-a-tool | `genai_tk/mcp/agent_tool.py` |
| Standalone script | `genai_tk/mcp/script_generator.py` |
| CLI | `genai_tk/mcp/cli_commands.py` |

## Agent Tool (`agent.enabled: true`)

`register_agent_tool()` bundles a server's resolved tools into a single
`run_<name>` MCP tool, built lazily and **cached** across calls (rebuilding a
sandbox-backed DeepAgent per call would be expensive). Key behaviors to keep
in mind when touching `genai_tk/mcp/agent_tool.py`:

- `agent.profile` resolves through `create_harness()` — any profile in the
  unified `agents:` dict works (`type: react`/`deep`/`custom` LangChain, or a
  DeerFlow profile), not just LangChain.
- Each call gets its own isolated `thread_id` (a fresh UUID) — never reuse a
  shared literal thread id, or concurrent MCP sessions bleed state into each
  other. Callers can pass back a previous `thread_id` to continue that
  conversation on the next call.
- The tool returns a structured `AgentToolResult` (`text`, `thread_id`,
  `error`), not a bare string — failures are captured as `error`, not raised.

## Change Workflow

1. Add `config/examples/tk_servers.yaml` config first when exposing existing genai-tk assets.
2. Use the same tool factory syntax as agent profile YAML.
3. Only add Python when adapting a new kind of callable/tool or changing server generation.
4. Verify `list`, `serve`, and `generate` paths if CLI behavior changes.

Use `config/mcp_servers.yaml` instead when configuring external MCP servers for agents to consume.

## Commands

```bash
uv run cli mcp list
uv run cli mcp generate --name <name> --output /tmp/<name>_server.py
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/mcp -q
```

## Avoid

- Do not duplicate tool registration logic between agents and MCP; share factory syntax.
- Do not require network services in MCP unit tests.
- Do not put secrets in generated scripts.
