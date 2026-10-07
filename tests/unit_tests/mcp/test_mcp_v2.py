"""Tests for MCP 2.0 features, Client(server) in-memory testing, and multi-transport support."""

from __future__ import annotations

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from genai_tk.mcp.config import MCPAgentConfig, MCPServerDefinition
from genai_tk.mcp.server_builder import build_mcp_server
from genai_tk.mcp.tool_adapter import register_tools


class CalculatorInput(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(default=0, description="Second integer")


@tool("calculator", args_schema=CalculatorInput)
def calculator(a: int, b: int = 0) -> int:
    """Calculate sum of two integers."""
    return a + b


def plain_multiply(x: int, y: int = 2) -> int:
    """Multiply two integers."""
    return x * y


@pytest.mark.asyncio
async def test_mcp2_client_with_in_memory_server():
    """Test MCP 2.0 Client interacting with an in-memory MCPServer."""
    from mcp import Client
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("test-calc", instructions="Calculator MCP 2.0 server")
    register_tools(server, [calculator, plain_multiply])

    async with Client(server) as client:
        # Verify server metadata
        assert client.server_info is not None
        assert client.server_info.name == "test-calc"

        # Verify tool listing
        tools_result = await client.list_tools()
        tool_names = [t.name for t in tools_result.tools]
        assert "calculator" in tool_names
        assert "plain_multiply" in tool_names

        # Verify tool schema has 'type': 'object'
        calc_tool = next(t for t in tools_result.tools if t.name == "calculator")
        schema = getattr(calc_tool, "input_schema", getattr(calc_tool, "inputSchema", {}))
        assert schema.get("type") == "object"
        assert "a" in schema.get("properties", {})

        # Verify calling structured tool
        res_calc = await client.call_tool("calculator", {"a": 15, "b": 27})
        assert not res_calc.is_error
        assert any("42" in getattr(c, "text", str(c)) for c in res_calc.content)

        # Verify calling plain callable tool
        res_mul = await client.call_tool("plain_multiply", {"x": 6, "y": 7})
        assert not res_mul.is_error
        assert any("42" in getattr(c, "text", str(c)) for c in res_mul.content)


@pytest.mark.asyncio
async def test_mcp2_server_builder_agent_tool():
    """Test building an MCP 2.0 server with an agent tool wrapper."""
    from mcp import Client

    defn = MCPServerDefinition(
        name="adhoc-agent-server",
        description="Server exposing ad-hoc agent",
        tools=[],
        agent=MCPAgentConfig(
            enabled=True,
            name="run_calc_agent",
            description="Agent that performs calculations",
        ),
    )
    server = build_mcp_server(defn)

    async with Client(server) as client:
        tools_result = await client.list_tools()
        names = [t.name for t in tools_result.tools]
        assert "run_calc_agent" in names

        agent_tool = next(t for t in tools_result.tools if t.name == "run_calc_agent")
        schema = getattr(agent_tool, "input_schema", getattr(agent_tool, "inputSchema", {}))
        assert schema.get("type") == "object"
        assert "query" in schema.get("properties", {})


def test_mcp2_remote_transport_config():
    """Test update_server_parameters with SSE and streamable-http URL configs."""
    from genai_tk.core.mcp_client import update_server_parameters

    sse_cfg = {
        "url": "http://localhost:8000/sse",
        "transport": "sse",
        "headers": {"Authorization": "Bearer test-token"},
    }
    processed_sse = update_server_parameters(sse_cfg)
    assert processed_sse["transport"] == "sse"
    assert processed_sse["url"] == "http://localhost:8000/sse"
    assert processed_sse["headers"]["Authorization"] == "Bearer test-token"

    http_cfg = {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable-http",
    }
    processed_http = update_server_parameters(http_cfg)
    assert processed_http["transport"] == "streamable-http"
    assert processed_http["url"] == "http://localhost:8000/mcp"
