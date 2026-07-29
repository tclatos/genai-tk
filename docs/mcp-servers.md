# Exposing genai-tk Assets as MCP Servers

The `genai_tk.mcp` package lets you expose LangChain tools and agents as
[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) servers using
only a YAML configuration file — no extra Python code required.

## Concepts

| Term | Meaning |
|---|---|
| **Server definition** | One entry in `config/examples/tk_servers.yaml`; maps to a single MCP server process |
| **Tool** | A LangChain tool factory resolved at startup and registered as an MCP tool |
| **Agent tool** | An optional ReAct / DeepAgent wrapper that bundles all resolved tools into a single `run_<name>` MCP tool |

## Configuration

Definitions live in `config/examples/tk_servers.yaml` under the key `mcp_expose_servers`.

```yaml
mcp_expose_servers:

  search:
    description: "Web search tools exposed as MCP"
    tools:
      - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_function
        verbose: false
    agent:
      enabled: true
      name: run_search_agent
      description: "Run a full ReAct web-search agent and return the final answer"
      # llm: gpt_41mini@openai   # override the LLM
      # profile: research        # use a langchain profile (by KEY)
```

The `tools` syntax is identical to `config/agents.yaml` — a `factory` key plus
any flat kwargs forwarded to the factory function.

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
MCP tool called `run_<name>` (configurable via `agent.name`). The tool's
harness is initialised lazily on the first call and cached across subsequent
calls (rebuilding a sandbox-backed DeepAgent per call would be expensive).

Use `agent.profile` to delegate to any profile in the unified `agents:` dict —
`type: deep`/`react`/`custom` (LangChain) or a DeerFlow profile, resolved via
`create_harness()`. Omit it to get a minimal ad-hoc ReAct agent over the
server's own resolved tools.

The tool returns a structured result — `{text, thread_id, error}` — rather
than a bare string. Each call gets its own isolated `thread_id` (a fresh UUID)
so concurrent MCP sessions never share conversation state; pass back a
previous call's `thread_id` explicitly to continue that conversation on the
next call.

## Adding a New Server

1. Add an entry to `config/examples/tk_servers.yaml`.
2. Run `uv run cli mcp list` to verify it appears.
3. Run `uv run cli mcp serve --name <name>` to start it.

No code changes needed unless you are writing a new tool factory.
