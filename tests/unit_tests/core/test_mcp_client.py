"""Unit tests for MCP client functionality (pytest style)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from mcp import StdioServerParameters

    from genai_tk.core.mcp_client import (
        dict_to_stdio_server_list,
        get_mcp_servers_dict,
        get_mcp_tools_info,
        update_server_parameters,
    )
except ImportError as e:
    pytest.skip(f"MCP feature not installed: {e}", allow_module_level=True)

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# update_server_parameters
# ---------------------------------------------------------------------------


def test_basic_server_parameters() -> None:
    config = {"command": "echo", "args": ["hello", "world"], "transport": "stdio"}

    result = update_server_parameters(config)

    assert result["command"] == "echo"
    assert result["args"] == ["hello", "world"]
    assert result["transport"] == "stdio"
    assert "PATH" in result["env"]


def test_uvx_command_alias() -> None:
    config = {"command": "uvx", "args": ["some-tool", "arg1"]}

    result = update_server_parameters(config)

    assert result["command"] == "uv"
    assert result["args"] == ["tool", "run", "some-tool", "arg1"]


def test_default_transport() -> None:
    config = {"command": "test", "args": []}

    result = update_server_parameters(config)

    assert result["transport"] == "stdio"


def test_environment_variables() -> None:
    config = {"command": "test", "args": [], "env": {"CUSTOM_VAR": "value"}}

    result = update_server_parameters(config)

    assert "PATH" in result["env"]
    assert result["env"]["CUSTOM_VAR"] == "value"


def test_removes_unused_keys() -> None:
    config = {
        "command": "test",
        "args": [],
        "description": "A test server",
        "example": "usage example",
        "disabled": False,
    }

    result = update_server_parameters(config)

    assert "description" not in result
    assert "example" not in result
    assert "disabled" not in result


# ---------------------------------------------------------------------------
# dict_to_stdio_server_list
# ---------------------------------------------------------------------------


def test_dict_to_stdio_server_list_empty() -> None:
    assert dict_to_stdio_server_list({}) == []


def test_dict_to_stdio_server_list_single() -> None:
    servers = {"test_server": {"command": "echo", "args": ["hello"], "transport": "stdio"}}

    result = dict_to_stdio_server_list(servers)

    assert len(result) == 1
    assert isinstance(result[0], StdioServerParameters)
    assert result[0].command == "echo"
    assert result[0].args == ["hello"]


def test_dict_to_stdio_server_list_multiple() -> None:
    servers = {
        "server1": {"command": "echo", "args": ["hello"], "transport": "stdio"},
        "server2": {"command": "cat", "args": ["file.txt"], "transport": "stdio"},
    }

    result = dict_to_stdio_server_list(servers)

    assert len(result) == 2
    assert all(isinstance(r, StdioServerParameters) for r in result)


# ---------------------------------------------------------------------------
# get_mcp_servers_dict
# ---------------------------------------------------------------------------


def _raw_config(servers: dict) -> object:
    return OmegaConf.create({"mcpServers": servers})


@patch("genai_tk.core.mcp_client.update_server_parameters")
@patch("genai_tk.core.mcp_client.get_raw_config")
def test_get_all_servers(mock_get_raw_config, mock_update) -> None:
    mock_get_raw_config.return_value = _raw_config(
        {
            "server1": {"command": "test1"},
            "server2": {"command": "test2", "disabled": False},
            "server3": {"command": "test3", "disabled": True},
        }
    )
    mock_update.side_effect = lambda x: x

    result = get_mcp_servers_dict()

    assert "server1" in result
    assert "server2" in result
    assert "server3" not in result
    mock_update.assert_called()


@patch("genai_tk.core.mcp_client.get_raw_config")
def test_filter_servers(mock_get_raw_config) -> None:
    mock_get_raw_config.return_value = _raw_config(
        {
            "server1": {"command": "test1"},
            "server2": {"command": "test2"},
        }
    )

    result = get_mcp_servers_dict(filter=["server1"])

    assert list(result.keys()) == ["server1"]


@patch("genai_tk.core.mcp_client.get_raw_config")
def test_missing_server_in_filter_raises(mock_get_raw_config) -> None:
    mock_get_raw_config.return_value = _raw_config({"server1": {"command": "test1"}})

    with pytest.raises(ValueError, match="nonexistent"):
        get_mcp_servers_dict(filter=["nonexistent"])


@patch("genai_tk.core.mcp_client.get_raw_config")
def test_disabled_server_in_filter_raises(mock_get_raw_config) -> None:
    mock_get_raw_config.return_value = _raw_config(
        {
            "server1": {"command": "test1"},
            "server2": {"command": "test2", "disabled": True},
        }
    )

    with pytest.raises(ValueError, match="disabled"):
        get_mcp_servers_dict(filter=["server2"])


@patch("genai_tk.core.mcp_client.update_server_parameters")
@patch("genai_tk.core.mcp_client.get_raw_config")
def test_server_configuration_error(mock_get_raw_config, mock_update) -> None:
    mock_get_raw_config.return_value = _raw_config(
        {
            "server1": {"command": "test1"},
            "server2": {"command": "test2"},
        }
    )
    mock_update.side_effect = [Exception("config error"), {"command": "fixed"}]

    result = get_mcp_servers_dict()

    assert len(result) == 1
    assert "server2" in result


# ---------------------------------------------------------------------------
# get_mcp_tools_info (async — MCP SDK is a true external boundary, kept mocked)
# ---------------------------------------------------------------------------


@patch("mcp.client.stdio.stdio_client")
@patch("mcp.ClientSession")
async def test_get_mcp_tools_info(mock_client_session, mock_stdio_client) -> None:
    mock_session = AsyncMock()
    mock_client_session.return_value.__aenter__.return_value = mock_session

    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool1.description = "First tool"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    mock_tool2.description = "Second tool"
    mock_tools_response = MagicMock()
    mock_tools_response.tools = [mock_tool1, mock_tool2]
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

    mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

    with patch("genai_tk.core.mcp_client.get_mcp_servers_dict") as mock_get_servers:
        mock_get_servers.return_value = {"test_server": {"command": "test", "args": []}}

        result = await get_mcp_tools_info()

    assert "test_server" in result
    assert len(result["test_server"]) == 2
    assert result["test_server"]["tool1"] == "First tool"
    assert result["test_server"]["tool2"] == "Second tool"
