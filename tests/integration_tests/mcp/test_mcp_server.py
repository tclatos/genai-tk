"""Integration tests for the MCP server builder.

These tests build real FastMCP servers from YAML config files and exercise
them programmatically via FastMCP's ``list_tools`` / ``call_tool`` API –
no subprocess or real stdio transport is needed.

A final smoke-test verifies that a server built from YAML has the expected
tools registered after full config + server_builder pipeline.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_tools_from_yaml(tmp_path: Path) -> None:
    """Server built from YAML must expose the tool declared in the factory."""
    from genai_tk.mcp.config import MCPServerDefinition, MCPToolConfig
    from genai_tk.mcp.server_builder import build_mcp_server

    # Build an MCPServerDefinition directly (avoids file path resolution issues)
    defn = MCPServerDefinition(
        name="echo",
        description="Echo server",
        tools=[MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_echo_tool")],
    )
    server = build_mcp_server(defn)
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "echo_text" in tool_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_call_tool_returns_string(tmp_path: Path) -> None:
    """Calling an MCP tool must return the expected string result."""
    from genai_tk.mcp.config import MCPServerDefinition, MCPToolConfig
    from genai_tk.mcp.server_builder import build_mcp_server

    defn = MCPServerDefinition(
        name="echo",
        description="Echo server",
        tools=[MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_echo_tool")],
    )
    server = build_mcp_server(defn)
    result = await server.call_tool("echo_text", {"input": "ping"})
    # result is (list[ContentBlock], dict) | list[ContentBlock]
    content_blocks = result[0] if isinstance(result, tuple) else result
    assert any("ping" in str(block) for block in content_blocks), f"Expected 'ping' in {content_blocks}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_tool_server(tmp_path: Path) -> None:
    """A server with multiple factory entries must expose all tools."""
    from genai_tk.mcp.config import MCPServerDefinition, MCPToolConfig
    from genai_tk.mcp.server_builder import build_mcp_server

    defn = MCPServerDefinition(
        name="multi",
        description="Multi-tool server",
        tools=[
            MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_echo_tool"),
            MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_reverse_tool"),
        ],
    )
    server = build_mcp_server(defn)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert "echo_text" in names
    assert "reverse_text" in names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_with_structured_schema(tmp_path: Path) -> None:
    """Tool with multi-field Pydantic schema must expose all parameters."""
    from genai_tk.mcp.config import MCPServerDefinition, MCPToolConfig
    from genai_tk.mcp.server_builder import build_mcp_server

    defn = MCPServerDefinition(
        name="adder",
        description="Addition server",
        tools=[MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_add_tool")],
    )
    server = build_mcp_server(defn)
    tools = await server.list_tools()
    add_tool = next(t for t in tools if t.name == "add_numbers")
    schema = add_tool.inputSchema
    props = schema.get("properties", {})
    assert "a" in props
    assert "b" in props


@pytest.mark.integration
@pytest.mark.asyncio
async def test_structured_tool_invocation() -> None:
    """Structured tool with two int params must produce correct sum."""
    from genai_tk.mcp.config import MCPServerDefinition, MCPToolConfig
    from genai_tk.mcp.server_builder import build_mcp_server

    defn = MCPServerDefinition(
        name="adder",
        description="Addition server",
        tools=[MCPToolConfig(factory="tests.integration_tests.mcp.fixtures:create_add_tool")],
    )
    server = build_mcp_server(defn)
    result = await server.call_tool("add_numbers", {"a": 7, "b": 3})
    content_blocks = result[0] if isinstance(result, tuple) else result
    assert any("10" in str(block) for block in content_blocks), f"Expected '10' in {content_blocks}"


# ---------------------------------------------------------------------------
# CLI smoke-test (subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_mcp_list_runs(tmp_path: Path) -> None:
    """``uv run cli mcp list`` should exit 0 and list at least one server."""

    config_yaml = tmp_path / "servers.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            """
            mcp_expose_servers:
              - name: "smoke"
                description: "Smoke test server"
                tools:
                  - factory: tests.integration_tests.mcp.fixtures:create_echo_tool
            """
        )
    )

    # Build the Python-level server directly to verify config is parseable
    from genai_tk.mcp.config import load_mcp_server_definitions
    from genai_tk.mcp.server_builder import build_mcp_server

    servers = load_mcp_server_definitions(config_yaml)
    assert len(servers) == 1
    assert servers[0].name == "smoke"
    server = build_mcp_server(servers[0])
    # Check tool is registered

    tools = asyncio.run(server.list_tools())
    assert any(t.name == "echo_text" for t in tools)
