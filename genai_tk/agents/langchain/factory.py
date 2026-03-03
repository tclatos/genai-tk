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
    BackendConfig,
    CheckpointerConfig,
    create_backend,
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

    # 7. Backend (deep agents only; ignored for react/custom with a warning)
    backend_cfg = profile.backend or BackendConfig(type="none")
    if backend_cfg.type != "none" and profile.type != "deep":
        logger.warning(
            f"Profile '{profile.name}' (type={profile.type}) has a backend configured "
            "but backends are only used by deep agents — ignoring."
        )
    backend = await create_backend(backend_cfg) if profile.type == "deep" else None

    # 8. Dispatch to engine
    if profile.type == "react":
        return _create_react_agent(model, all_tools, middlewares, checkpointer, profile)

    if profile.type == "deep":
        return await _create_deep_agent(
            model, all_tools, checkpointer, profile, middlewares=middlewares, backend=backend
        )

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


async def _create_deep_agent(
    model: Any,
    tools: list[BaseTool],
    checkpointer: Any,
    profile: AgentProfileConfig,
    *,
    middlewares: list | None = None,
    backend: Any = None,
) -> Any:
    """Build a deep agent using deepagents.create_deep_agent.

    When ``skill_directories`` are configured, skills are loaded via
    deepagents' native ``SkillsMiddleware`` using progressive disclosure:
    the LLM sees skill metadata and reads full content on demand via file
    tools.  An explicit ``system_prompt`` in the YAML profile is passed
    through unchanged.
    """
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend

    skill_dirs = _resolve_skill_dirs(profile.skill_directories)

    # When skills are present, ensure we have a FilesystemBackend so the
    # SkillsMiddleware can scan the directories and the LLM can read files.
    if skill_dirs and backend is None:
        from genai_tk.utils.config_mngr import global_config

        project_root = global_config().paths.project
        backend = FilesystemBackend(root_dir=project_root, virtual_mode=True)

    # Convert absolute skill paths to backend-relative paths so that
    # deepagents' SkillsMiddleware can resolve them via the backend.
    relative_skills: list[str] | None = None
    if skill_dirs and backend is not None:
        backend_root = str(getattr(backend, "cwd", ""))
        relative_skills = []
        for d in skill_dirs:
            rel = __import__("os").path.relpath(d, backend_root)
            relative_skills.append(rel)

    logger.info(
        f"Deep agent '{profile.name}': planning={profile.enable_planning}, "
        f"filesystem={profile.enable_file_system}, skill_dirs={len(skill_dirs)}, "
        f"backend={type(backend).__name__ if backend is not None else 'none'}"
    )

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=profile.system_prompt or None,
        skills=relative_skills,
        middleware=middlewares or [],
        checkpointer=checkpointer,
        backend=backend,
    )
    # Attach the backend so callers can stop it during cleanup
    agent._backend = backend  # type: ignore[attr-defined]
    return agent


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


def _load_skills_as_prompt(skill_dirs: list[str]) -> str | None:
    """Read all SKILL.md files from the given directories and return them concatenated.

    Files are discovered recursively and loaded in sorted path order so the
    result is deterministic.  The front-matter block (``---`` … ``---``) is
    stripped before concatenation.  A Rich table is printed to the terminal
    listing each loaded skill.

    Args:
        skill_dirs: List of resolved directory paths to search.

    Returns:
        Concatenated skill content separated by ``---``, or None if no skills
        were found.

    Example:
    ```
        prompt = _load_skills_as_prompt(["/skills/my-agent"])
    ```
    """
    import re
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    sections: list[str] = []
    skill_names: list[tuple[str, str]] = []  # (name, description)

    for skill_dir in skill_dirs:
        for skill_file in sorted(Path(skill_dir).rglob("SKILL.md")):
            try:
                raw = skill_file.read_text().strip()
                # Extract name/description from front-matter if present
                fm_name = skill_file.parent.name
                fm_desc = ""
                fm_match = re.match(r"^---\n(.*?)\n---\n", raw, flags=re.DOTALL)
                if fm_match:
                    for line in fm_match.group(1).splitlines():
                        if line.startswith("name:"):
                            fm_name = line.split(":", 1)[1].strip()
                        elif line.startswith("description:"):
                            fm_desc = line.split(":", 1)[1].strip()
                content = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()
                if content:
                    sections.append(content)
                    skill_names.append((fm_name, fm_desc))
            except Exception as exc:
                logger.warning(f"Could not read skill file '{skill_file}': {exc}")

    if not sections:
        return None

    # Print Rich table
    console = Console()
    table = Table(title="Skills loaded as system prompt", show_lines=False, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Skill", style="bold cyan")
    table.add_column("Description", style="dim")
    for i, (name, desc) in enumerate(skill_names, 1):
        table.add_row(str(i), name, desc or "—")
    console.print(table)

    logger.info(f"Loaded {len(sections)} skill(s) as system prompt")
    return "\n\n---\n\n".join(sections)


def _resolve_skill_dirs(skill_directories: list[str]) -> list[str]:
    """Resolve ``${...}`` variable interpolation in skill directory paths.

    Uses OmegaConf so that paths like ``${paths.project}/skills/my-skill``
    are resolved correctly even when the variable is not the entire string.
    """
    if not skill_directories:
        return []

    from omegaconf import OmegaConf

    from genai_tk.utils.config_mngr import global_config

    try:
        cfg = OmegaConf.create({"dirs": skill_directories})
        merged = OmegaConf.merge(global_config().root, cfg)
        dirs = OmegaConf.to_container(merged, resolve=True)["dirs"]  # type: ignore[index]
        resolved = [str(d) for d in dirs]
    except Exception:
        # Fall back to unresolved paths so the agent still starts
        resolved = list(skill_directories)

    existing = [d for d in resolved if __import__("os").path.isdir(d)]
    missing = [d for d in resolved if d not in existing]
    if missing:
        logger.warning(f"Skill directories not found (will be ignored): {missing}")
    return existing
