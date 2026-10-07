# Exposing genai-tk Assets as MCP Servers (MCP 2.0)

The `genai_tk.mcp` package lets you expose LangChain tools, Python callables, and agents as
[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) 2.0 servers using
only a YAML configuration file — no extra Python boilerplate required.

It supports both **server-side exposure** (over `stdio`, `sse`, or `streamable-http`) and **client-side consumption**
(with `open_mcp_client`, `Client`, or `MultiServerMCPClient`).

## Concepts

| Term | Meaning |
|---|---|
| **Server definition** | One entry in `config/tk_servers.yaml` (or `config/examples/tk_servers.yaml`); maps to an MCP server process |
| **Tool** | A LangChain `BaseTool` or plain Python callable resolved at startup and registered as an MCP tool |
| **Agent tool** | An optional ReAct or DeepAgent wrapper that bundles tools into a single `run_<name>` / `ask_<name>` MCP tool |
| **Transports** | `stdio` (subprocess standard I/O), `sse` (Server-Sent Events), or `streamable-http` (HTTP streaming) |

## MCP 2.0 Protocol & Compatibility

`genai-tk` is built on **MCP 2.0+** (`mcp>=2.0.0` and `mcp-types`):
- **`MCPServer` core**: The server orchestrator uses the official MCP 2.0 `MCPServer` class (`mcp.server.mcpserver`).
- **Snake_case models**: All wire models use standard Python snake_case attributes (`input_schema`, `is_error`, `structured_content`, `next_cursor`).
- **Compatibility layer (`genai_tk.mcp.compat`)**: Automatic shims ensure seamless backward compatibility with legacy consumers such as `langchain-mcp-adapters` and MCP 1.x clients:
  - Aliases `FastMCP` $\leftrightarrow$ `MCPServer` and module hierarchy `mcp.server.fastmcp.*`.
  - Bridges wire model properties (`tool.inputSchema`, `result.isError`, `result.structuredContent`).
  - Adapts `ClientSession.list_tools` and related listing methods for legacy `cursor` parameter callers.
- **Stdio stream isolation**: Stdout redirection during startup protects stdio streams against rogue native C/Rust prints (such as BAML log initialization) that would otherwise corrupt the JSON-RPC wire.

## Configuration

Definitions live in `config/tk_servers.yaml` (fallback to `config/examples/tk_servers.yaml`) under the key `mcp_expose_servers`:

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
      # profile: research        # use an agent profile (by key)

  docgraph-tools:
    description: "Document Graph navigation tools"
    tools:
      - factory: genai_graph.agent.docgraph_agent.create_document_graph_tools_from_config
```

The `tools` syntax supports:
- Factory functions returning `BaseTool`, a list of `BaseTool`, or plain callable functions.
- Parameters forwarded directly to the factory (or via nested `config:`).

OmegaConf variables (`${paths.project}`) are resolved against the global config
before the definitions are loaded.

### Project Servers in `config/mcp_servers.yaml`

To consume project-local servers from within agent profiles or other tools, register them under `mcpProjectServers`:

```yaml
mcpProjectServers:
  docgraph-tools:
    description: "Document Graph navigation tools"
  docgraph-agent:
    description: "Document Graph deep agent"
```

Each project server is automatically served via `uv --project <root> run cli mcpserver start --name <name>`.

## CLI Commands

```bash
# List all configured servers
uv run cli mcpserver list

# Start a server (stdio transport, default)
uv run cli mcpserver start --name search
# Or using the 'serve' alias:
uv run cli mcpserver serve --name search

# Start with SSE or Streamable-HTTP transport with custom host and port
uv run cli mcpserver start --name search --transport sse --port 8001
uv run cli mcpserver start --name search --transport streamable-http --host 0.0.0.0 --port 8000

# Inspect tools or call a tool on any server via CLI
uv run cli core mcp-call search
uv run cli core mcp-call search --tool internet_search --tool-args '{"query": "LangChain news"}'

# Generate a standalone Python script (for use with uvx or Claude Desktop)
uv run cli mcpserver generate --name search --output server_search.py
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

## Agent Tool (`agent.enabled: true`)

When `agent.enabled: true` is set, all resolved tools are bundled into a single
MCP tool called `run_<name>` or custom `agent.name`. The tool's
harness is initialised lazily on the first call and cached across subsequent
calls (rebuilding a sandbox-backed DeepAgent per call would be expensive).

- Use `agent.profile` to delegate to any profile in the unified `agents:` dict —
  `type: deep`/`react`/`custom` (LangChain) or a DeerFlow profile, resolved via
  `create_harness()`.
- Supports custom harness factories or specialized profiles like `docgraph` automatically.
- Omit `agent.profile` to get a minimal ad-hoc ReAct agent over the server's own resolved tools.

The tool returns a structured result — `{text, thread_id, error}` — rather
than a bare string. Each call gets its own isolated `thread_id` (a fresh UUID)
so concurrent MCP sessions never share conversation state; pass back a
previous call's `thread_id` explicitly to continue that conversation on the
next call.

## Client Usage

To connect to any configured MCP server from Python:

```python
from genai_tk.core.mcp_client import open_mcp_client, get_mcp_servers_dict

servers = get_mcp_servers_dict(filter=["docgraph-tools"])
server_desc = servers["docgraph-tools"]

async with open_mcp_client(server_desc) as client:
    tools = await client.list_tools()
    result = await client.call_tool("list_documents", {})
    print(result.content)
```

## Adding a New Server

1. Add an entry to `config/tk_servers.yaml`.
2. Run `uv run cli mcpserver list` to verify it appears.
3. Run `uv run cli mcpserver start --name <name>` to start it.

No code changes needed unless you are writing a new tool factory.
