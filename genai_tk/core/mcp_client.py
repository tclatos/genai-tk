"""Utilities for interacting with MCP (Multi-Component Platform) servers.

This module provides functionality to:
- Retrieve and configure MCP servers from application configuration
- Create and manage MCP client connections
- Execute queries using MCP tools with LangChain agents
- Validate and process server configurations
- Run MCP agents with custom queries and server filters

The main components are:
- get_mcp_servers_dict: Retrieves and processes MCP server configurations
- update_server_parameters: Processes individual server configurations
- mcp_agent_runner: Executes queries using MCP tools with a ReAct agent
- call_react_agent: Convenience function for running MCP agents with streaming output

Example usage:
```python
# Get all configured MCP servers
servers = get_mcp_servers_dict()

# Run a query using specific MCP servers
await call_react_agent("What's the weather in Toulouse?", mcp_server_filter=["weather"])
```
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from devtools import debug  # noqa: F401
from dotenv import load_dotenv
from langchain_core.language_models.base import LanguageModelOutput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

import genai_tk.mcp.compat  # noqa: F401
from genai_tk.config_mgmt.config_mngr import get_raw_config, paths_config

load_dotenv()


class McpServerConfig(BaseModel):
    """Pydantic schema for a single MCP server entry from YAML configuration.

    Covers both ``mcpServers`` (external) and ``mcpProjectServers`` (project-local)
    entries. Unknown keys are ignored so future YAML additions stay backward
    compatible.
    """

    command: str | None = Field(None, description="Executable to launch the server")
    args: list[str] = Field(default_factory=list, description="Arguments passed to the command")
    url: str | None = Field(None, description="Endpoint URL for SSE / streamable-http")
    transport: Literal["stdio", "sse", "streamable-http", "http"] = Field("stdio", description="Transport protocol")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers for remote transports")
    env: dict[str, str] = Field(default_factory=dict, description="Additional environment variables")
    disabled: bool = Field(False, description="Set true to skip this server")
    description: str | None = Field(None, description="Human-readable description (ignored at runtime)")
    example: str | None = Field(None, description="Usage example (ignored at runtime)")

    model_config = ConfigDict(extra="ignore")


def update_server_parameters(server_config: dict) -> dict:
    """Process individual MCP server configuration dictionary.

    Handles command aliases, environment variables, and validation.
    Ensures required parameters are present and properly formatted.

    Args:
        server_config: Raw server configuration dictionary.

    Returns:
        Processed server parameters dictionary ready for server instantiation
    """
    from genai_tk.config_mgmt.config_exceptions import yaml_config_validation

    with yaml_config_validation(context="MCP server config"):
        cfg = McpServerConfig.model_validate(server_config)

    if cfg.url or cfg.transport in ("sse", "streamable-http", "http"):
        transport = cfg.transport if cfg.transport != "stdio" else "streamable-http"
        res: dict[str, Any] = {
            "url": cfg.url,
            "transport": transport,
            "headers": cfg.headers,
            "env": cfg.env,
        }
        return res

    if not cfg.command:
        raise ValueError("MCP server configuration must provide either 'command' or 'url'.")

    # Resolve uvx alias to 'uv tool run'
    command = cfg.command
    args = list(cfg.args)
    if command == "uvx":
        command = "uv"
        args = ["tool", "run"] + args

    desc: dict = {
        "command": command,
        "args": args,
        "transport": "stdio",
        "env": {"PATH": os.environ.get("PATH", "")} | cfg.env,
    }
    from mcp import StdioServerParameters  # noqa: PLC0415

    clean_desc = {k: v for k, v in desc.items() if k in ("command", "args", "env", "cwd")}
    _ = StdioServerParameters(**clean_desc)  # validate against MCP library schema
    return desc


@asynccontextmanager
async def open_mcp_client(server_desc: dict) -> AsyncIterator[Any]:
    """Async context manager to connect to an MCP server across any transport.

    Supports stdio (via command/args) and network transports (sse, streamable-http, http)
    using ClientSession or Client.
    """
    url = server_desc.get("url")
    transport = server_desc.get("transport", "stdio")

    if url or transport in ("sse", "streamable-http", "http"):
        from mcp import Client  # noqa: PLC0415

        headers = server_desc.get("headers") or None
        target = url or f"http://{server_desc.get('host', '127.0.0.1')}:{server_desc.get('port', 8000)}"
        async with Client(target, headers=headers) as client:
            yield client
    else:
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415

        clean_params = {
            "command": server_desc["command"],
            "args": server_desc.get("args", []),
            "env": server_desc.get("env"),
            "cwd": server_desc.get("cwd"),
        }
        params = StdioServerParameters(**clean_params)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def get_mcp_tools_info(filter: list[str] | None = None) -> dict:
    """Get all tools from MCP servers with their names and descriptions."""
    servers = get_mcp_servers_dict(filter)
    tools_info = {}
    for server_name, param_desc in servers.items():
        debug(server_name)
        if not param_desc.get("disabled", False):
            try:
                async with open_mcp_client(param_desc) as client:
                    tools_result = await client.list_tools()
                    tool_list = getattr(tools_result, "tools", tools_result)
                    tools_info[server_name] = {
                        tool.name: getattr(tool, "description", "") or "" for tool in tool_list
                    }
            except Exception as e:
                logger.warning("Error fetching tools for MCP server {}: {}", server_name, e)
    return tools_info


async def get_mcp_tools_with_schema(filter: list[str] | None = None) -> dict[str, list]:
    """Get all tools from MCP servers including their input schemas.

    Returns:
        Dict mapping server name to list of MCP Tool objects (with .name, .description, .input_schema).
    """
    servers = get_mcp_servers_dict(filter)
    result: dict[str, list] = {}
    for server_name, param_desc in servers.items():
        debug(server_name)
        if not param_desc.get("disabled", False):
            try:
                async with open_mcp_client(param_desc) as client:
                    tools_result = await client.list_tools()
                    tool_list = getattr(tools_result, "tools", tools_result)
                    result[server_name] = list(tool_list)
            except Exception as e:
                logger.warning("Error fetching tools with schema for MCP server {}: {}", server_name, e)
    return result


async def get_mcp_prompts(filter: list[str] | None = None) -> dict:
    """Get all prompts from MCP servers with their names and descriptions."""
    servers = get_mcp_servers_dict(filter)
    prompts_info = {}
    for server_name, param_desc in servers.items():
        if not param_desc.get("disabled", False):
            try:
                async with open_mcp_client(param_desc) as client:
                    prompts_result = await client.list_prompts()
                    prompt_list = getattr(prompts_result, "prompts", prompts_result)
                    prompts_info[server_name] = {
                        p.name: getattr(p, "description", "") or "" for p in prompt_list
                    }
            except Exception as e:
                logger.warning("Error fetching prompts for MCP server {}: {}", server_name, e)
    return prompts_info


def get_mcp_servers_dict(filter: list[str] | None = None) -> dict:
    """Retrieve configured MCP servers from application configuration.

    Combines servers from two config sections:
    - ``mcpServers``: external MCP servers (npm, uvx, …)
    - ``mcpProjectServers``: project-defined servers exposed via
      ``cli mcpserver start --name <name>`` (declared in ``config/examples/tk_servers.yaml``).

    Processes each entry, handling command aliases and environment variables,
    then validates server parameters.

    Args:
        filter: List of server names to include. If None, all servers are returned.

    Returns:
        Dictionary of server names to their configuration parameters

    Example:
    ```python
    servers = get_mcp_servers_dict()
    # {'chinook': {'command': 'uv', 'args': ['--project', '...', 'run', 'cli', 'mcpserver', 'start', '--name', 'chinook'], ...}}

    servers = get_mcp_servers_dict(filter=["chinook"])
    ```
    """
    from omegaconf import OmegaConf

    raw = get_raw_config()
    servers_node = raw.get("mcpServers", {})
    servers: dict[str, dict] = {}
    if servers_node:
        servers_container = OmegaConf.to_container(servers_node, resolve=True)
        if isinstance(servers_container, dict):
            servers = {str(name): cfg for name, cfg in servers_container.items() if isinstance(cfg, dict)}

    # Merge in project-defined servers declared under ``mcpProjectServers``.
    # Each entry is served via ``uv --project <root> run cli mcp serve --name <name>``.
    try:
        project_servers_node = raw.get("mcpProjectServers", {})
        project_servers_cfg: dict[str, dict] = {}
        if project_servers_node:
            project_servers_container = OmegaConf.to_container(project_servers_node, resolve=True)
            if isinstance(project_servers_container, dict):
                project_servers_cfg = {
                    str(name): cfg for name, cfg in project_servers_container.items() if isinstance(cfg, dict)
                }
        project_root = str(paths_config().project)
        for pname, pcfg in project_servers_cfg.items():
            if (pcfg or {}).get("disabled", False):
                servers[pname] = {"disabled": True}
            else:
                servers[pname] = {
                    "command": "uv",
                    "args": ["--project", project_root, "run", "cli", "mcpserver", "start", "--name", pname],
                    "transport": "stdio",
                }
    except Exception:
        pass  # mcpProjectServers is optional

    all_servers = servers
    disabled_servers = {name for name, config in all_servers.items() if config.get("disabled", False)}
    enabled_servers = {name: config for name, config in all_servers.items() if name not in disabled_servers}

    if filter is not None:
        disabled_in_filter = sorted(name for name in filter if name in disabled_servers)
        not_found_in_filter = sorted(name for name in filter if name not in all_servers)
        if disabled_in_filter or not_found_in_filter:
            details: list[str] = []
            if disabled_in_filter:
                details.append(f"disabled: {', '.join(disabled_in_filter)}")
            if not_found_in_filter:
                details.append(f"not found: {', '.join(not_found_in_filter)}")

            available_enabled_servers = sorted(enabled_servers.keys())
            available_txt = ", ".join(available_enabled_servers) if available_enabled_servers else "none"
            raise ValueError(
                f"Invalid MCP server filter ({'; '.join(details)}). Available enabled servers: {available_txt}."
            )

    result_dict = {}
    for name, desc in enabled_servers.items():
        if filter is None or name in filter:
            try:
                result_dict[name] = update_server_parameters(desc)
            except Exception as e:
                logger.warning("Skipping MCP server {} due to configuration error: {}", name, str(e))
    return result_dict


def dict_to_stdio_server_list(param_list: dict) -> list:
    """Convert a dictionary of server parameters to StdioServerParameters objects.

    Args:
        param_list: Dictionary where keys are server names and values are
                   server parameter dictionaries

    Returns:
        List of StdioServerParameters instances ready for MCP client creation

    Example:
    ```python
    servers = {"weather": {"command": "uv", "args": ["tool", "run", "weathermcp"]}}
    server_params = dict_to_stdio_server_list(servers)
    # [StdioServerParameters(command='uv', args=['tool', 'run', 'weathermcp'], ...)]
    ```
    """
    from mcp import StdioServerParameters  # noqa: PLC0415

    valid_keys = {"command", "args", "env", "cwd", "encoding", "encoding_error_handler"}
    return [
        StdioServerParameters(**{k: v for k, v in desc.items() if k in valid_keys})
        for name, desc in param_list.items()
        if desc.get("command")
    ]


async def mcp_agent_runner(
    model: BaseChatModel, servers: list | dict, prompt: str, config: RunnableConfig | None = None
) -> LanguageModelOutput | None:
    """Execute a query using MCP tools with a ReAct agent.

    Creates a ReAct agent with MCP tools and processes the query.

    Args:
        model: The language model to use for the agent
        servers: List of StdioServerParameters or server dictionary
        prompt: The input query to process
        config: Optional RunnableConfig for the agent execution

    Returns:
        The final response content from the agent, or None if no response

    Example:
    ```python
    model = get_llm()
    servers = get_mcp_servers_dict()
    response = await mcp_agent_runner(model, servers, "What's the weather?")
    ```
    """
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if config is None:
        config = {}

    if isinstance(servers, dict):
        servers_dict = servers
    else:
        servers_dict = {}
        for idx, s in enumerate(servers):
            if hasattr(s, "command"):
                servers_dict[f"server_{idx}"] = {
                    "command": s.command,
                    "args": s.args,
                    "env": s.env,
                    "transport": "stdio",
                }
            elif isinstance(s, dict):
                servers_dict[s.get("name", f"server_{idx}")] = s

    client = MultiServerMCPClient(servers_dict)
    tools = await client.get_tools()

    memory = MemorySaver() if config.get("thread_id") else None
    agent_executor = create_agent(model, tools, checkpointer=memory)

    result = await agent_executor.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config,
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    examples = [
        "what's the weather in Toulouse ? ",
        "list files in current directory",
        "connect to atos.net and get recent news",
    ]
    # This module is now primarily focused on low-level MCP client utilities.
    # Interactive and rich-agent helpers live in
    # `genai_tk.agents.langchain_agent_rich`.
