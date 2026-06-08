---
name: genai-tk-mcp-servers
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
| Standalone script | `genai_tk/mcp/script_generator.py` |
| CLI | `genai_tk/mcp/cli_commands.py` |

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
