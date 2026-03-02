"""Low-level deep-agent utilities for GenAI Toolkit.

High-level profile loading and agent creation have moved to:
- ``genai_tk.agents.langchain.config`` — Pydantic config models and loading
- ``genai_tk.agents.langchain.factory`` — unified agent factory

This module retains only the minimal runtime helpers:
- MCP server validation
- Agent execution wrappers (run / stream)
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

# Type alias for the compiled graph (CompiledStateGraph from LangGraph)
type DeepAgent = Any  # Actually CompiledStateGraph


# ============================================================================
# Exceptions
# ============================================================================


class DeepAgentError(Exception):
    """Base exception for deep agent errors."""


class MCPServerNotFoundError(DeepAgentError):
    """Raised when MCP server names are not found in configuration."""

    def __init__(self, invalid_servers: list[str], available_servers: list[str] | None = None):
        self.invalid_servers = invalid_servers
        self.available_servers = available_servers
        message = f"MCP server(s) not found: {', '.join(invalid_servers)}"
        if available_servers:
            message += f". Available servers: {', '.join(available_servers)}"
        super().__init__(message)


# ============================================================================
# Validation
# ============================================================================


def validate_mcp_servers(server_names: list[str]) -> None:
    """Validate that MCP server names exist in the current configuration.

    Args:
        server_names: List of MCP server names to validate.
    """
    from genai_tk.utils.config_mngr import global_config

    try:
        mcp_servers = global_config().get_dict("mcpServers")
        available_names = [name for name, config in mcp_servers.items() if not config.get("disabled", False)]
        invalid = [name for name in server_names if name not in available_names]
        if invalid:
            raise MCPServerNotFoundError(invalid, available_names)
    except MCPServerNotFoundError:
        raise
    except Exception as e:
        logger.warning(f"Could not validate MCP servers: {e}")


# ============================================================================
# Agent execution
# ============================================================================


async def run_deep_agent(
    agent: DeepAgent,
    input_message: str,
    thread_id: str = "default",
    stream: bool = False,
    on_chunk: Any | None = None,
) -> dict[str, Any]:
    """Run a deep agent with the given input.

    Args:
        agent: Deep agent instance (CompiledStateGraph).
        input_message: User input message.
        thread_id: Thread ID for conversation memory (requires a checkpointer).
        stream: Whether to stream the response chunks.
        on_chunk: Optional callback invoked per streaming chunk.

    Returns:
        Dict with a ``messages`` key containing the conversation.

    Example:
    ```python
    result = await run_deep_agent(agent, "Research quantum computing")
    print(result["messages"][-1].content)
    ```
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [{"role": "user", "content": input_message}]}

    if stream:
        chunks: list[Any] = []
        async for chunk in agent.astream(input_data, config=config):
            chunks.append(chunk)
            if on_chunk:
                on_chunk(chunk)
        return chunks[-1] if chunks else {"messages": []}

    return await agent.ainvoke(input_data, config=config)


def run_deep_agent_sync(
    agent: DeepAgent,
    input_message: str,
    thread_id: str = "default",
    stream: bool = False,
    on_chunk: Any | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for ``run_deep_agent``.

    See ``run_deep_agent`` for full documentation.
    """
    return asyncio.run(
        run_deep_agent(
            agent=agent,
            input_message=input_message,
            thread_id=thread_id,
            stream=stream,
            on_chunk=on_chunk,
        )
    )
