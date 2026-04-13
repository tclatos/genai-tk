"""Expose a LangChain-based agent as a single MCP tool named ``run_<name>``.

The agent is built lazily (on first call) so the MCP server starts instantly
and only incurs the LLM/MCP-server bootstrap cost when a client calls the tool.

Example:
    ```python
    from mcp.server.fastmcp import FastMCP
    from genai_tk.mcp.config import MCPAgentConfig
    from genai_tk.mcp.agent_tool import register_agent_tool

    server = FastMCP("my-server")
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

from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger
from mcp.server.fastmcp import FastMCP

from genai_tk.mcp.config import MCPAgentConfig


def register_agent_tool(
    server: FastMCP,
    agent_cfg: MCPAgentConfig,
    extra_tools: list[BaseTool] | None = None,
) -> None:
    """Register a wrapped agent as a single ``query: str → str`` MCP tool.

    The agent instance is created on the first invocation (lazy init) so that
    the MCP server process starts immediately without waiting for heavy LLM /
    MCP-server connections to initialise.

    Args:
        server: FastMCP server instance to register the tool on.
        agent_cfg: Agent configuration (name, description, profile, llm).
        extra_tools: Additional LangChain tools passed to the agent on top of
            what the profile declares (can be None).
    """
    assert isinstance(agent_cfg, MCPAgentConfig)

    _agent_cache: dict[str, Any] = {}

    async def _invoke_agent(query: str) -> str:
        """Run the agent and return the final text answer."""
        if "agent" not in _agent_cache:
            _agent_cache["agent"] = await _build_agent(agent_cfg, extra_tools or [])
            logger.info("Agent '{}' initialised.", agent_cfg.name)

        agent = _agent_cache["agent"]
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "mcp_default"}},
        )
        messages = result.get("messages", [])
        if not messages:
            return "No response from agent."
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        return content if isinstance(content, str) else str(content)

    _invoke_agent.__name__ = agent_cfg.name
    _invoke_agent.__doc__ = agent_cfg.description

    server.add_tool(_invoke_agent, name=agent_cfg.name, description=agent_cfg.description)
    logger.debug("Registered agent MCP tool: {!r}", agent_cfg.name)


async def _build_agent(agent_cfg: MCPAgentConfig, extra_tools: list[BaseTool]) -> Any:
    """Instantiate an agent from a profile (or a minimal react agent).

    Args:
        agent_cfg: Agent configuration.
        extra_tools: Extra tools to pass alongside profile tools.

    Returns:
        A compiled LangGraph agent.
    """
    from langgraph.checkpoint.memory import MemorySaver

    if agent_cfg.profile:
        from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
        from genai_tk.agents.langchain.factory import create_langchain_agent

        cfg = load_unified_config()
        profile = resolve_profile(cfg, agent_cfg.profile)
        return await create_langchain_agent(
            profile,
            llm_override=agent_cfg.llm,
            extra_tools=extra_tools or None,
            force_memory_checkpointer=True,
        )

    # No profile: build a minimal prebuilt ReAct agent with the provided tools
    from langgraph.prebuilt import create_react_agent

    from genai_tk.core.llm_factory import get_llm

    llm = get_llm(agent_cfg.llm) if agent_cfg.llm else get_llm()
    return create_react_agent(llm, extra_tools, checkpointer=MemorySaver())
