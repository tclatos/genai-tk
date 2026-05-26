---
name: add-mcp-server
description: Add a Model Context Protocol (MCP) server to a genai-tk project — config, credential management, and validation.
tags: [mcp, tools, configuration]
version: "1.0"
---

# Add an MCP Server

MCP (Model Context Protocol) servers expose tools to agents over a standardized JSON-RPC interface.
genai-tk manages them through `config/mcp_servers.yaml`.

## Step 1: Find or Implement the MCP Server

**Use an existing server:**
- `@modelcontextprotocol/server-filesystem` — file system access
- `@modelcontextprotocol/server-brave-search` — web search
- `@modelcontextprotocol/server-github` — GitHub API
- Browse: https://github.com/modelcontextprotocol/servers

**Implement your own:**  
See `examples/mcp_server/` for a Python MCP server skeleton.

## Step 2: Add to mcp_servers.yaml

Edit `config/mcp_servers.yaml`:

```yaml
mcp_servers:
  my_server:                        # ← key used in agent profiles
    transport: stdio                # stdio | sse
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-brave-search"
    env:
      BRAVE_API_KEY: ${oc.env:BRAVE_API_KEY}  # env var interpolation
    enabled: true

  my_python_server:
    transport: stdio
    command: uv
    args:
      - run
      - python
      - examples/mcp_server/server.py
    enabled: true
```

## Step 3: Reference in Agent Profile

In `config/agents/langchain.yaml`:

```yaml
langchain_agents:
  my_agent:
    mcp_servers:
      - my_server
      - my_python_server
```

## Step 4: Add Credentials

Add to `.env` (never commit this file):
```
BRAVE_API_KEY=bsk-xxxxx
```

Add to `.env.example` (commit this):
```
BRAVE_API_KEY=              # get from https://api.search.brave.com/
```

## Step 5: Validate

```bash
cli mcp list                    # list configured servers
cli mcp test my_server          # verify connection
cli agents langchain -p my_agent "search for: Python MCP"
```

## Transport Types

| Transport | Use when | Notes |
|-----------|----------|-------|
| `stdio` | local process or Docker | default, most reliable |
| `sse` | remote/HTTP server | requires running server URL |

For `sse` transport:
```yaml
my_server:
  transport: sse
  url: http://localhost:8080/sse
```

## Code Map

| Concern | Path |
|---------|------|
| MCP config | `config/mcp_servers.yaml` |
| MCP CLI commands | `genai_tk/cli/commands_mcp.py` |
| MCP docs | `docs/mcp-servers.md` |
| Example server | `examples/mcp_server/server.py` |
| Agent profiles | `config/agents/langchain.yaml` |
