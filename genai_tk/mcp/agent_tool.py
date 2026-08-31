"""Expose a LangChain-based agent as a single MCP tool named ``run_<name>``.

The agent harness is built lazily (on first call) so the MCP server starts
instantly and only incurs the LLM/MCP-server bootstrap cost when a client
calls the tool. The harness is cached and reused across calls (avoiding a
rebuild — especially costly for sandbox-backed DeepAgents); each call gets
its own isolated conversation thread unless the caller explicitly passes one
back to continue a prior turn.

Example:
    ```python
    from mcp.server.mcpserver import MCPServer
    from genai_tk.mcp.config import MCPAgentConfig
    from genai_tk.mcp.agent_tool import register_agent_tool

    server = MCPServer("my-server")
    cfg = MCPAgentConfig(
        enabled=True,
        name="run_research_agent",
        description="Run the Research agent",
        profile="Research",
    )
    register_agent_tool(server, cfg, extra_tools=[])
    ```
"""

from __future__ import annotations

import uuid

from langchain_core.tools import BaseTool
from loguru import logger

try:
    from mcp.server.mcpserver import MCPServer
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]
from pydantic import BaseModel

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.mcp.config import MCPAgentConfig


class AgentToolResult(BaseModel):
    """Structured result returned by an MCP agent-as-a-tool call."""

    text: str = ""
    thread_id: str = ""
    error: str | None = None


def register_agent_tool(
    server: MCPServer,
    agent_cfg: MCPAgentConfig,
    extra_tools: list[BaseTool] | None = None,
) -> None:
    """Register a wrapped agent as a single MCP tool.

    The harness instance is created on the first invocation (lazy init) so
    that the MCP server process starts immediately without waiting for heavy
    LLM / MCP-server connections to initialise.

    Args:
        server: MCPServer instance to register the tool on.
        agent_cfg: Agent configuration (name, description, profile, llm).
        extra_tools: Additional LangChain tools passed to the agent on top of
            what the profile declares (can be None).
    """
    assert isinstance(agent_cfg, MCPAgentConfig)

    _harness_cache: dict[str, BaseHarness] = {}

    async def _invoke_agent(query: str, thread_id: str | None = None) -> AgentToolResult:
        """Run the agent and return the final text answer.

        Args:
            query: User query for the agent to answer.
            thread_id: Conversation thread to continue; omit for a fresh,
                isolated conversation (never shares state with other calls).
        """
        if "harness" not in _harness_cache:
            _harness_cache["harness"] = await _build_harness(agent_cfg, extra_tools or [])
            logger.info("Agent '{}' initialised.", agent_cfg.name)

        harness = _harness_cache["harness"]
        tid = thread_id or f"mcp-{uuid.uuid4().hex}"
        try:
            text = await harness.arun(query, thread_id=tid)
        except Exception as exc:
            logger.opt(exception=True).warning(f"Agent '{agent_cfg.name}' run failed: {exc}")
            return AgentToolResult(thread_id=tid, error=str(exc))
        return AgentToolResult(text=text, thread_id=tid)

    _invoke_agent.__name__ = agent_cfg.name
    _invoke_agent.__doc__ = agent_cfg.description

    server.add_tool(_invoke_agent, name=agent_cfg.name, description=agent_cfg.description)
    logger.debug("Registered agent MCP tool: {!r}", agent_cfg.name)


async def _build_harness(agent_cfg: MCPAgentConfig, extra_tools: list[BaseTool]) -> BaseHarness:
    """Build a harness from a profile (or a minimal adhoc react agent).

    Args:
        agent_cfg: Agent configuration.
        extra_tools: Extra tools to pass alongside profile tools.

    Returns:
        A ready-to-stream :class:`BaseHarness` (langchain or DeerFlow).
    """
    if agent_cfg.profile:
        from genai_tk.agents.harness.registry import create_harness

        return create_harness(
            agent_cfg.profile,
            llm_override=agent_cfg.llm,
            force_memory_checkpointer=True,
            extra_tools=extra_tools or None,
        )

    # No profile: build a minimal adhoc react harness with the provided tools.
    from genai_tk.agents.harness.langchain_harness import LangChainHarness
    from genai_tk.agents.langchain.config import AgentProfileConfig, CheckpointerConfig

    profile = AgentProfileConfig(
        name=agent_cfg.name,
        type="react",
        llm=agent_cfg.llm,
        checkpointer=CheckpointerConfig(type="none"),
    )
    return LangChainHarness(profile, force_memory_checkpointer=True, extra_tools=extra_tools)
