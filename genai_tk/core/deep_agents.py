"""Deep Agents wrapper for GenAI Toolkit.

Creates and configures Deep Agents using GenAI Toolkit's LLM factory and tool system,
while leveraging deepagents' built-in capabilities (planning, file system, subagents, skills).

Deep agents use deepagents v0.4+ which natively accepts LangChain BaseTool instances,
provides middleware for planning (write_todos), file system tools (read_file, write_file, edit_file),
subagent spawning, and progressive skill disclosure.

See: https://docs.langchain.com/oss/python/deepagents/overview
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain.tools import BaseTool
from loguru import logger

# Type alias for the compiled graph (CompiledStateGraph from LangGraph)
type DeepAgent = Any  # Actually CompiledStateGraph


# ============================================================================
# Custom Exceptions
# ============================================================================


class DeepAgentError(Exception):
    """Base exception for deep agent errors."""


class ProfileNotFoundError(DeepAgentError):
    """Raised when a profile name is not found in configuration."""

    def __init__(self, profile_name: str, available_profiles: list[str]):
        self.profile_name = profile_name
        self.available_profiles = available_profiles
        super().__init__(f"Profile '{profile_name}' not found. Available profiles: {', '.join(available_profiles)}")


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
# Configuration Model
# ============================================================================


@dataclass
class DeepAgentProfileConfig:
    """Configuration for a deep agent profile.

    Loaded from config/agents/deepagents.yaml.
    """

    name: str
    description: str = ""
    system_prompt: str | None = None
    llm: str | None = None  # LLM identifier (ID like 'gpt_41mini@openai' or tag like 'fast_model')
    tools: list[BaseTool] = field(default_factory=list)
    tool_configs: list[dict[str, Any]] = field(default_factory=list)  # Raw tool configs from YAML
    mcp_servers: list[str] = field(default_factory=list)
    skill_directories: list[str] = field(default_factory=list)  # Directories with SKILL.md files
    enable_planning: bool = True  # Enable write_todos planning tool
    enable_file_system: bool = True  # Enable file system tools (read_file, write_file, etc.)
    subagents: list[dict[str, Any]] = field(default_factory=list)  # Subagent configurations
    features: list[str] = field(default_factory=list)  # Feature badges for UI
    examples: list[str] = field(default_factory=list)  # Example queries


# ============================================================================
# Config Loading
# ============================================================================


def load_deep_agent_profiles(
    config_path: str | None = None,
) -> list[DeepAgentProfileConfig]:
    """Load deep agent profiles from YAML config.

    Example YAML structure:
        deep_agent_profiles:
          - name: "Research"
            description: "Expert researcher with web search"
            system_prompt: "You are an expert researcher..."
            tools:
              - factory: genai_tk.tools.langchain.search_tools_factory:create_search_function
            mcp_servers:
              - tavily-mcp
            skill_directories:
              - ${paths.project}/skills
            enable_planning: true
            enable_file_system: true

    Args:
        config_path: Path to the deepagents.yaml config file. If None, uses global config.

    Returns:
        List of DeepAgentProfileConfig instances

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or empty
    """
    from genai_tk.utils.config_mngr import global_config

    if config_path is None:
        # Use config manager to get the proper path
        config_dir = global_config().get_dir_path("paths.config")
        config_path = str(config_dir / "agents" / "deepagents.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Deep agent config not found at {path}. Please create the config file or specify a valid path."
        )

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if not raw or "deep_agent_profiles" not in raw:
        raise ValueError(f"Config file {path} is missing 'deep_agent_profiles' section. Please check the file format.")

    profiles = []
    for entry in raw.get("deep_agent_profiles", []):
        raw_tools = entry.get("tools", [])
        profile = DeepAgentProfileConfig(
            name=entry.get("name", "Unnamed"),
            description=entry.get("description", ""),
            system_prompt=entry.get("system_prompt"),
            llm=entry.get("llm"),
            tool_configs=raw_tools,
            mcp_servers=entry.get("mcp_servers", []),
            skill_directories=entry.get("skill_directories", []),
            enable_planning=entry.get("enable_planning", True),
            enable_file_system=entry.get("enable_file_system", True),
            subagents=entry.get("subagents", []),
            features=entry.get("features", []),
            examples=entry.get("examples", []),
        )
        profiles.append(profile)

    logger.info(f"Loaded {len(profiles)} deep agent profiles from {config_path}")
    return profiles


def get_default_profile_name() -> str:
    """Get the default deep agent profile name from config.

    Returns:
        Default profile name (defaults to 'Research' if not configured)
    """
    from genai_tk.utils.config_mngr import global_config

    try:
        return global_config().get("deep_agents.default_profile", "Research")
    except Exception:
        return "Research"


# ============================================================================
# Validation Functions
# ============================================================================


def validate_profile_name(profile_name: str, available_profiles: list[DeepAgentProfileConfig]) -> None:
    """Validate that a profile name exists in the configuration.

    Args:
        profile_name: Name of the profile to validate
        available_profiles: List of available profile configurations

    Raises:
        ProfileNotFoundError: If profile name is not found
    """
    profile_names = [p.name for p in available_profiles]
    if profile_name not in profile_names:
        raise ProfileNotFoundError(profile_name, profile_names)


def validate_mcp_servers(server_names: list[str]) -> None:
    """Validate that MCP server names exist in configuration.

    Args:
        server_names: List of MCP server names to validate

    Raises:
        MCPServerNotFoundError: If any server names are invalid
    """
    from genai_tk.utils.config_mngr import global_config

    try:
        mcp_servers = global_config().get_dict("mcpServers")
        available_names = [name for name, config in mcp_servers.items() if not config.get("disabled", False)]
        invalid = [name for name in server_names if name not in available_names]

        if invalid:
            raise MCPServerNotFoundError(invalid, available_names)
    except Exception as e:
        if isinstance(e, MCPServerNotFoundError):
            raise
        logger.warning(f"Could not validate MCP servers: {e}")


def validate_llm_identifier(llm_id: str) -> None:
    """Validate that an LLM identifier can be resolved.

    Args:
        llm_id: LLM identifier to validate (e.g., 'gpt_41mini@openai' or 'fast_model')

    Raises:
        ValueError: If LLM identifier cannot be resolved
    """
    from genai_tk.core.llm_factory import LlmFactory

    try:
        LlmFactory.resolve_llm_identifier_safe(llm_id)
    except Exception as e:
        raise ValueError(f"Invalid LLM identifier '{llm_id}': {e}") from e


# ============================================================================
# Tool Resolution
# ============================================================================


def resolve_tools_from_config(tool_configs: list[dict[str, Any]]) -> list[BaseTool]:
    """Resolve LangChain tools from configuration specs.

    Supports same patterns as React and Deer Flow agents:
    - factory: module:function - call factory function
    - class: module:Class - instantiate class
    - function: module:func - wrap function as tool

    Args:
        tool_configs: List of tool configuration dicts

    Returns:
        List of instantiated LangChain BaseTool instances

    Example:
        tool_configs = [
            {"factory": "genai_tk.tools.langchain.search_tools_factory:create_search_function"},
            {"class": "langchain_community.tools:WikipediaQueryRun"},
        ]
        tools = resolve_tools_from_config(tool_configs)
    """
    from genai_tk.tools.langchain.shared_config_loader import process_langchain_tools_from_config

    return process_langchain_tools_from_config(tool_configs)


# ============================================================================
# Agent Creation
# ============================================================================


async def create_deep_agent_from_profile(
    profile: DeepAgentProfileConfig,
    llm: Any | None = None,
    extra_tools: list[BaseTool] | None = None,
    extra_mcp_servers: list[str] | None = None,
    checkpointer: Any | None = None,
) -> DeepAgent:
    """Create a deep agent from a profile configuration.

    This is the main entry point for creating deep agents. It:
    1. Resolves the LLM (from profile or parameter)
    2. Resolves tools from profile config
    3. Loads MCP tools from specified servers
    4. Creates the deep agent using deepagents.create_deep_agent()

    Args:
        profile: Deep agent profile configuration
        llm: Optional LLM override. If None, uses profile.llm or default from config.
        extra_tools: Optional additional tools to add
        extra_mcp_servers: Optional additional MCP server names
        checkpointer: Optional LangGraph checkpointer for conversation memory

    Returns:
        DeepAgent (CompiledStateGraph) instance

    Example:
        profiles = load_deep_agent_profiles()
        profile = next(p for p in profiles if p.name == "Research")
        agent = await create_deep_agent_from_profile(profile)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Research AI"}]})
    """
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from genai_tk.core.llm_factory import get_llm
    from genai_tk.core.mcp_client import get_mcp_servers_dict

    # 1. Resolve LLM
    if llm is None:
        if profile.llm:
            llm = get_llm(profile.llm)
        else:
            # Use default LLM from config
            llm = get_llm()

    # 2. Resolve profile tools
    profile_tools = resolve_tools_from_config(profile.tool_configs)

    # 3. Load MCP tools
    mcp_tools: list[BaseTool] = []
    all_mcp_servers = list(profile.mcp_servers)
    if extra_mcp_servers:
        all_mcp_servers.extend(extra_mcp_servers)

    if all_mcp_servers:
        # Validate MCP server names
        validate_mcp_servers(all_mcp_servers)

        # Get MCP server configs
        mcp_servers_dict = get_mcp_servers_dict()
        selected_servers = {name: mcp_servers_dict[name] for name in all_mcp_servers if name in mcp_servers_dict}

        if selected_servers:
            try:
                client = MultiServerMCPClient(selected_servers)
                mcp_tools = await client.get_tools()
                logger.info(f"Loaded {len(mcp_tools)} tools from {len(selected_servers)} MCP server(s)")
            except Exception as e:
                logger.warning(f"Failed to load MCP tools: {e}")

    # 4. Combine all tools
    all_tools = profile_tools + mcp_tools
    if extra_tools:
        all_tools.extend(extra_tools)

    logger.info(
        f"Creating deep agent '{profile.name}' with {len(all_tools)} tools, "
        f"planning={profile.enable_planning}, filesystem={profile.enable_file_system}"
    )

    # 5. Prepare skill directories (resolve variables)
    skill_dirs: list[str] = []
    if profile.skill_directories:
        from genai_tk.utils.config_mngr import global_config

        for skill_dir in profile.skill_directories:
            # Replace ${paths.project} or similar variables
            if "${" in skill_dir:
                try:
                    resolved = str(global_config().get_dir_path(skill_dir.replace("${", "").replace("}", "")))
                    skill_dirs.append(resolved)
                except Exception:
                    skill_dirs.append(skill_dir)
            else:
                skill_dirs.append(skill_dir)

    # 6. Create the deep agent using deepagents v0.4+ API
    # The new API natively accepts BaseTool instances (no conversion needed)
    # Note: subagents parameter requires SubAgent objects - not yet implemented
    agent = create_deep_agent(
        model=llm,
        tools=all_tools,
        system_prompt=profile.system_prompt,
        skills=skill_dirs or None,
        checkpointer=checkpointer,
        # Note: enable_planning and enable_file_system are controlled by middleware
        # In v0.4, these are enabled by default. To disable, we'd need to pass custom middleware.
    )

    return agent


def create_deep_agent_from_profile_sync(
    profile: DeepAgentProfileConfig,
    llm: Any | None = None,
    extra_tools: list[BaseTool] | None = None,
    extra_mcp_servers: list[str] | None = None,
    checkpointer: Any | None = None,
) -> DeepAgent:
    """Synchronous wrapper for create_deep_agent_from_profile.

    See create_deep_agent_from_profile for full documentation.
    """
    return asyncio.run(
        create_deep_agent_from_profile(
            profile=profile,
            llm=llm,
            extra_tools=extra_tools,
            extra_mcp_servers=extra_mcp_servers,
            checkpointer=checkpointer,
        )
    )


# ============================================================================
# Agent Execution
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
        agent: Deep agent instance (CompiledStateGraph)
        input_message: User input message
        thread_id: Thread ID for conversation memory (if checkpointer is used)
        stream: Whether to stream the response
        on_chunk: Optional callback for streaming chunks

    Returns:
        Dict with 'messages' key containing the conversation, and optionally 'structured_response'

    Example:
        agent = await create_deep_agent_from_profile(profile)
        result = await run_deep_agent(agent, "Research quantum computing")
        print(result["messages"][-1].content)
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [{"role": "user", "content": input_message}]}

    if stream:
        # Stream the response
        chunks = []
        async for chunk in agent.astream(input_data, config=config):
            chunks.append(chunk)
            if on_chunk:
                on_chunk(chunk)

        # Return the final state
        if chunks:
            return chunks[-1]
        return {"messages": []}
    else:
        # Non-streaming invocation
        result = await agent.ainvoke(input_data, config=config)
        return result


def run_deep_agent_sync(
    agent: DeepAgent,
    input_message: str,
    thread_id: str = "default",
    stream: bool = False,
    on_chunk: Any | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for run_deep_agent.

    See run_deep_agent for full documentation.
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


# ============================================================================
# Convenience Functions
# ============================================================================


def list_deep_agent_profiles(config_path: str | None = None) -> list[str]:
    """List available deep agent profile names.

    Args:
        config_path: Optional path to config file

    Returns:
        List of profile names
    """
    profiles = load_deep_agent_profiles(config_path)
    return [p.name for p in profiles]


def get_deep_agent_profile(profile_name: str, config_path: str | None = None) -> DeepAgentProfileConfig:
    """Get a specific deep agent profile by name.

    Args:
        profile_name: Name of the profile to retrieve
        config_path: Optional path to config file

    Returns:
        DeepAgentProfileConfig instance

    Raises:
        ProfileNotFoundError: If profile name is not found
    """
    profiles = load_deep_agent_profiles(config_path)
    validate_profile_name(profile_name, profiles)
    return next(p for p in profiles if p.name == profile_name)
