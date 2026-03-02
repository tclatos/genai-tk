"""Unified agent factory for all LangChain-based agent types.

Single entry point that dispatches to the correct engine based on
``AgentProfileConfig.type`` (react | deep | custom).

Example:
```python
from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
from genai_tk.agents.langchain.factory import create_langchain_agent

config = load_unified_config()
profile = resolve_profile(config, "Research")
agent = await create_langchain_agent(profile)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Research AI"}]})
```
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.agents.langchain.config import (
    AgentProfileConfig,
    CheckpointerConfig,
    create_checkpointer,
    instantiate_middlewares,
)
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.mcp_client import get_mcp_servers_dict
from genai_tk.tools.langchain.shared_config_loader import process_langchain_tools_from_config


async def create_langchain_agent(
    profile: AgentProfileConfig,
    llm_override: str | None = None,
    extra_tools: list[BaseTool] | None = None,
    extra_mcp_servers: list[str] | None = None,
    force_memory_checkpointer: bool = False,
) -> Any:
    """Create an agent from a resolved profile configuration.

    Dispatches to the correct engine based on ``profile.type``:
    - ``react`` → ``langchain.agents.create_agent``
    - ``deep`` → ``deepagents.create_deep_agent``
    - ``custom`` → ``genai_tk.extra.graphs.custom_react_agent.create_custom_react_agent``

    Args:
        profile: Resolved agent profile (defaults already merged in).
        llm_override: LLM identifier that takes precedence over ``profile.llm``.
        extra_tools: Additional tools appended after profile tools.
        extra_mcp_servers: Additional MCP server names appended to profile servers.
        force_memory_checkpointer: When True, always use ``MemorySaver`` even if the
            profile specifies ``checkpointer.type: none``. Use for ``--chat`` mode.

    Returns:
        A compiled LangGraph agent (``CompiledStateGraph`` or ``Pregel``).
    """
    # 1. Resolve LLM
    llm_id = llm_override or profile.llm
    model = get_llm(llm=llm_id)

    # 2. Resolve tools from profile
    profile_tools = process_langchain_tools_from_config(profile.tools, llm=llm_id)

    # 3. Load MCP tools
    all_mcp_servers = list(profile.mcp_servers)
    if extra_mcp_servers:
        all_mcp_servers.extend(extra_mcp_servers)

    mcp_tools: list[BaseTool] = []
    mcp_client = None
    if all_mcp_servers:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        mcp_servers_dict = get_mcp_servers_dict(all_mcp_servers)
        if mcp_servers_dict:
            mcp_client = MultiServerMCPClient(mcp_servers_dict)
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"Loaded {len(mcp_tools)} tools from {len(mcp_servers_dict)} MCP server(s)")

    # 4. Combine all tools
    all_tools: list[BaseTool] = profile_tools + mcp_tools
    if extra_tools:
        all_tools.extend(extra_tools)

    logger.info(f"Creating '{profile.name}' agent (type={profile.type}) with {len(all_tools)} tools")

    # 5. Checkpointer
    checkpointer_cfg = profile.checkpointer or CheckpointerConfig(type="none")
    checkpointer = create_checkpointer(checkpointer_cfg, force_memory=force_memory_checkpointer)

    # 6. Middleware
    middleware_cfgs = profile.middlewares or []
    middlewares = instantiate_middlewares(middleware_cfgs, profile.type)

    # 7. Dispatch to engine
    if profile.type == "react":
        return _create_react_agent(model, all_tools, middlewares, checkpointer, profile)

    if profile.type == "deep":
        return await _create_deep_agent(model, all_tools, checkpointer, profile)

    if profile.type == "custom":
        return _create_custom_agent(model, all_tools, checkpointer)

    raise ValueError(f"Unknown agent type: {profile.type!r}")


# ============================================================================
# Engine-specific builders
# ============================================================================


def _create_react_agent(
    model: Any, tools: list[BaseTool], middlewares: list, checkpointer: Any, profile: AgentProfileConfig
) -> Any:
    """Build a standard LangChain prebuilt ReAct agent."""
    from langchain.agents import create_agent

    kwargs: dict[str, Any] = {"model": model, "tools": tools, "middleware": middlewares}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    system_prompt = profile.system_prompt or profile.pre_prompt
    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    return create_agent(**kwargs)


async def _create_deep_agent(model: Any, tools: list[BaseTool], checkpointer: Any, profile: AgentProfileConfig) -> Any:
    """Build a deep agent using deepagents.create_deep_agent."""
    from deepagents import create_deep_agent

    skill_dirs = _resolve_skill_dirs(profile.skill_directories)

    logger.info(
        f"Deep agent '{profile.name}': planning={profile.enable_planning}, "
        f"filesystem={profile.enable_file_system}, skills={len(skill_dirs)}"
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=profile.system_prompt,
        skills=skill_dirs or None,
        checkpointer=checkpointer,
    )


def _create_custom_agent(model: Any, tools: list[BaseTool], checkpointer: Any) -> Any:
    """Build the custom Functional-API ReAct agent."""
    from langgraph.checkpoint.memory import MemorySaver

    from genai_tk.extra.graphs.custom_react_agent import create_custom_react_agent

    # custom agent requires a checkpointer
    cp = checkpointer if checkpointer is not None else MemorySaver()
    return create_custom_react_agent(model=model, tools=tools, checkpointer=cp)


# ============================================================================
# Helpers
# ============================================================================


def _resolve_skill_dirs(skill_directories: list[str]) -> list[str]:
    """Resolve ``${...}`` variable interpolation in skill directory paths."""
    if not skill_directories:
        return []

    from genai_tk.utils.config_mngr import global_config

    resolved: list[str] = []
    for skill_dir in skill_directories:
        if "${" in skill_dir:
            try:
                key = skill_dir.strip().lstrip("${").rstrip("}")
                resolved.append(str(global_config().get_dir_path(key)))
            except Exception:
                resolved.append(skill_dir)
        else:
            resolved.append(skill_dir)
    return resolved
