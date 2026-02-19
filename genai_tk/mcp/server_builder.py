"""Build and run FastMCP servers from MCPServerDefinition configuration.

This is the central orchestrator: it loads a definition, resolves tools from
factories, registers tool wrappers, optionally registers the agent-as-a-tool,
and either returns the server or runs it (blocking).

Example:
    ```python
    # Build and serve (blocks on stdio)
    from genai_tk.mcp.server_builder import serve
    serve("search")

    # Or get the server object for testing
    from genai_tk.mcp.server_builder import build_mcp_server
    from genai_tk.mcp.config import get_mcp_server_definition

    defn = get_mcp_server_definition("chinook")
    server = build_mcp_server(defn)
    ```
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from mcp.server.fastmcp import FastMCP

from genai_tk.mcp.config import MCPServerDefinition, get_mcp_server_definition, load_mcp_server_definitions
from genai_tk.mcp.tool_adapter import register_tools, resolve_tools_from_config


def build_mcp_server(definition: MCPServerDefinition) -> FastMCP:
    """Construct a FastMCP server from an MCPServerDefinition.

    Steps:
    1. Instantiate a ``FastMCP`` server with the definition's name/description.
    2. Resolve LangChain tools from all ``tools`` factory entries.
    3. Register each tool as an MCP tool.
    4. If ``agent`` is configured and enabled, register the agent-as-a-tool.

    Args:
        definition: Server definition loaded from YAML.

    Returns:
        Configured FastMCP server (not yet running).

    Example:
        ```python
        defn = get_mcp_server_definition("search")
        server = build_mcp_server(defn)
        server.run()  # stdio
        ```
    """
    server = FastMCP(definition.name, instructions=definition.description or None)

    # Step 1 – resolve LangChain tools from factory configs
    raw_tool_configs = [{"factory": t.factory, **t.factory_kwargs()} for t in definition.tools]
    lc_tools = resolve_tools_from_config(raw_tool_configs)
    logger.info(f"[{definition.name}] resolved {len(lc_tools)} LangChain tool(s)")

    # Step 2 – register individual tools as MCP tools
    register_tools(server, lc_tools)

    # Step 3 – optionally register the agent-as-a-tool
    if definition.agent and definition.agent.enabled:
        from genai_tk.mcp.agent_tool import register_agent_tool

        register_agent_tool(server, definition.agent, extra_tools=lc_tools)

    return server


def serve(
    name: str,
    config_path: Path | str | None = None,
    transport: str = "stdio",
) -> None:
    """Load a server definition by name and serve it (blocking).

    Args:
        name: Server name as declared in ``mcp_expose_servers`` YAML key.
        config_path: Path to the YAML config file; auto-detected if None.
        transport: MCP transport. ``'stdio'`` (default), ``'sse'``, or
            ``'streamable-http'``.

    Example:
        ```python
        from genai_tk.mcp.server_builder import serve
        serve("search")                        # stdio – used by Claude Desktop etc.
        serve("search", transport="sse")       # HTTP SSE
        ```
    """
    definition = get_mcp_server_definition(name, config_path)
    server = build_mcp_server(definition)
    logger.info(f"Starting MCP server '{name}' over {transport} transport …")
    server.run(transport=transport)  # type: ignore[arg-type]


def list_servers(config_path: Path | str | None = None) -> list[MCPServerDefinition]:
    """Return all server definitions from the config file.

    Args:
        config_path: Optional path override.

    Returns:
        List of MCPServerDefinition instances.
    """
    return load_mcp_server_definitions(config_path)
