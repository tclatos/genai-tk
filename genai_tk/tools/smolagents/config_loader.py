"""Shared configuration loader for SmolAgent demos.

Loads agent configurations from a YAML file or directory
(``config/basic/agents/smolagents.yaml`` or ``config/basic/agents/smolagents/``)
using OmegaConf for ``${...}`` interpolation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import BaseTool as LangChainBaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from smolagents import Tool as SmolAgentTool

from genai_tk.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec, tool_spec_from_dict
from genai_tk.utils.config_mngr import import_from_qualified, load_yaml_configs, paths_config

# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class SmolagentsAgentConfig(BaseModel):
    """Configuration class for CodeAct Agent demonstrations.

    This class defines the structure for setting up different demo scenarios
    including available tools, MCP servers, and example prompts.
    """

    name: str
    tool_specs: List[Union[ClassToolSpec, FunctionToolSpec, FactoryToolSpec]] = Field(
        default_factory=list, description="Tool specifications"
    )
    tools: List[Any] = Field(default_factory=list, description="Instantiated tool objects (set at runtime)")
    mcp_servers: List[str] = Field(default_factory=list, description="MCP server names")
    examples: List[str] = Field(default_factory=list, description="Example prompts")
    authorized_imports: List[str] = Field(default_factory=list, description="Authorized imports for execution")
    pre_prompt: Optional[str] = Field(None, description="Pre-prompt to include with exemplary behavior")

    model_config = ConfigDict(arbitrary_types_allowed=True)


def instantiate_tools_from_specs(
    tool_specs: List[Union[ClassToolSpec, FunctionToolSpec, FactoryToolSpec]],
) -> List[Any]:
    """Instantiate tool objects from tool specifications.

    Args:
        tool_specs: List of ToolSpec instances

    Returns:
        List of instantiated tool objects
    """
    tools = []

    for spec in tool_specs:
        try:
            if isinstance(spec, ClassToolSpec):
                tool_class = import_from_qualified(spec.tool_class)
                tool_instance = tool_class(**spec.extra_params)
                tools.append(tool_instance)
            elif isinstance(spec, FunctionToolSpec):
                tool_func = import_from_qualified(spec.function)
                tools.append(tool_func)
            elif isinstance(spec, FactoryToolSpec):
                factory_func = import_from_qualified(spec.factory)
                tool_result = factory_func(**spec.extra_params)

                if isinstance(tool_result, list):
                    tools.extend(tool_result)
                else:
                    tools.append(tool_result)
        except ModuleNotFoundError as ex:
            missing_module = str(ex).split("'")[1] if "'" in str(ex) else str(ex)
            logger.warning(f"Skipping tool: missing optional dependency '{missing_module}'")
        except (ImportError, AttributeError) as ex:
            logger.warning(f"Skipping tool {spec}: {ex}")
        except Exception as ex:
            logger.warning(f"Failed to instantiate tool from {spec}: {ex}")

    return tools


def convert_langchain_tools(tools: List[Any]) -> List[Any]:
    """Convert LangChain BaseTool instances to SmolAgent Tools.

    Args:
        tools: List of tool instances that may include LangChain tools

    Returns:
        List of tools with LangChain tools converted to SmolAgent tools
    """
    converted_tools = []
    for tool in tools:
        if isinstance(tool, LangChainBaseTool):
            converted_tools.append(SmolAgentTool.from_langchain(tool))
        else:
            converted_tools.append(tool)
    return converted_tools


def _load_smolagents_raw() -> list[dict[str, Any]]:
    """Load raw smolagents config entries from file or directory."""
    agents_dir = paths_config().config / "agents"
    dir_path = agents_dir / "smolagents"
    path: Path = dir_path if dir_path.is_dir() else agents_dir / "smolagents.yaml"
    return load_yaml_configs(path, "smolagents_codeact")  # type: ignore[return-value]


def load_smolagent_demo_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration for a specific demo by name.

    Args:
        config_name: Name of the configuration to load.

    Returns:
        Dictionary containing the demo configuration, or ``None`` if not found.
    """
    try:
        for demo_config in _load_smolagents_raw():
            if demo_config.get("name", "").lower() == config_name.lower():
                return demo_config
        return None
    except Exception as ex:
        logger.error(f"Failed to load demo config '{config_name}': {ex}")
        return None


def load_all_demos_from_config() -> List[SmolagentsAgentConfig]:
    """Load and configure all SmolAgent demonstration scenarios.

    Returns:
        List of configured ``SmolagentsAgentConfig`` objects.
    """
    try:
        demos_config = _load_smolagents_raw()
        result = []

        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            authorized_imports = demo_config.get("authorized_imports", [])
            pre_prompt = demo_config.get("pre_prompt")

            # Parse tool specifications from config
            tool_specs = []
            raw_tools_config = demo_config.get("tools", [])
            if raw_tools_config:
                for tool_cfg in raw_tools_config:
                    spec = tool_spec_from_dict(tool_cfg.copy())
                    if spec:
                        tool_specs.append(spec)

            # Instantiate tools from specifications
            raw_tools = instantiate_tools_from_specs(tool_specs)
            converted_tools = convert_langchain_tools(raw_tools)

            demo = SmolagentsAgentConfig(
                name=name,
                tool_specs=tool_specs,
                tools=converted_tools,
                mcp_servers=mcp_servers,
                examples=examples,
                authorized_imports=authorized_imports,
                pre_prompt=pre_prompt,
            )
            result.append(demo)

        return result
    except Exception as ex:
        logger.exception(f"Failed to load demo configurations: {ex}")
        return []


def create_demo_from_config(config_name: str) -> Optional[SmolagentsAgentConfig]:
    """Create a single SmolagentsAgentConfig from configuration.

    Args:
        config_name: Name of the configuration to load

    Returns:
        SmolagentsAgentConfig instance or None if not found
    """
    demo_config = load_smolagent_demo_config(config_name)
    if not demo_config:
        return None

    name = demo_config.get("name", "")
    examples = demo_config.get("examples", [])
    mcp_servers = demo_config.get("mcp_servers", [])
    authorized_imports = demo_config.get("authorized_imports", [])
    pre_prompt = demo_config.get("pre_prompt")

    # Parse tool specifications from config
    tool_specs = []
    raw_tools_config = demo_config.get("tools", [])
    if raw_tools_config:
        for tool_cfg in raw_tools_config:
            spec = tool_spec_from_dict(tool_cfg.copy())
            if spec:
                tool_specs.append(spec)

    # Instantiate tools from specifications
    raw_tools = instantiate_tools_from_specs(tool_specs)
    converted_tools = convert_langchain_tools(raw_tools)

    return SmolagentsAgentConfig(
        name=name,
        tool_specs=tool_specs,
        tools=converted_tools,
        mcp_servers=mcp_servers,
        examples=examples,
        authorized_imports=authorized_imports,
        pre_prompt=pre_prompt,
    )
