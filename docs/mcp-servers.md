# Exposing genai-tk Assets as MCP Servers

The `genai_tk.mcp` package lets you expose LangChain tools and agents as
[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) servers using
only a YAML configuration file — no extra Python code required.

## Concepts

| Term | Meaning |
|---|---|
| **Server definition** | One entry in `config/mcp/servers.yaml`; maps to a single MCP server process |
| **Tool** | A LangChain tool factory resolved at startup and registered as an MCP tool |
| **Agent tool** | An optional ReAct / DeepAgent wrapper that bundles all resolved tools into a single `run_<name>` MCP tool |

## Configuration

Definitions live in `config/mcp/servers.yaml` under the key `mcp_expose_servers`.

```yaml
mcp_expose_servers:

  - name: "search"
    description: "Web search tools exposed as MCP"
    tools:
      - factory: genai_tk.tools.langchain.search_tools_factory:create_search_function
        verbose: false
    agent:
      enabled: true
      name: run_search_agent
      description: "Run a full ReAct web-search agent and return the final answer"
      # llm: gpt_41mini@openai   # override the LLM
      # profile: Research        # use a deepagents.yaml profile
```

The `tools` syntax is identical to `langchain.yaml` — a `factory` key plus any
flat kwargs forwarded to the factory function.

OmegaConf variables (`${paths.project}`) are resolved against the global config
before the definitions are loaded.

## CLI Commands

```bash
# List all configured servers
uv run cli mcp list

# Start a server (stdio transport, default)
uv run cli mcp serve --name search

# Start with SSE or Streamable-HTTP transport
uv run cli mcp serve --name search --transport sse

# Generate a standalone Python script (for use with uvx or Claude Desktop)
uv run cli mcp generate --name search --output server_search.py
```

## Standalone Scripts

`generate` produces a self-contained script that can be referenced directly in
an MCP client configuration:

```json
{
  "mcpServers": {
    "search": {
      "command": "uv",
      "args": ["run", "server_search.py"]
    }
  }
}
```

## Agent Tool

When `agent.enabled: true` is set, all resolved tools are bundled into a single
MCP tool called `run_<name>` (configurable via `agent.name`). The agent is
initialised lazily on the first call.

Use `agent.profile` to delegate to a full DeepAgent profile defined in
`deepagents.yaml`.  Omit it to get a minimal ReAct agent.

## Adding a New Server

1. Add an entry to `config/mcp/servers.yaml`.
2. Run `uv run cli mcp list` to verify it appears.
3. Run `uv run cli mcp serve --name <name>` to start it.

No code changes needed unless you are writing a new tool factory.
