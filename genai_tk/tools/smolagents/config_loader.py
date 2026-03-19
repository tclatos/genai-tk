"""Shared configuration loader for SmolAgent demos.

Loads agent configurations from a YAML file or directory
(``config/basic/agents/smolagents.yaml`` or ``config/basic/agents/smolagents/``)
using OmegaConf for ``${...}`` interpolation.
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool as LangChainBaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from smolagents import Tool as SmolAgentTool

from genai_tk.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec, ToolSpec
from genai_tk.utils.config_mngr import import_from_qualified, load_yaml_configs, paths_config

# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class SmolagentsAgentConfig(BaseModel):
    """Configuration for a single SmolAgent demonstration scenario."""

    name: str
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    authorized_imports: list[str] = Field(default_factory=list)
    pre_prompt: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def instantiate_tools_from_specs(tool_specs: list[ToolSpec]) -> list[Any]:
    """Instantiate tool objects from tool specifications.

    Args:
        tool_specs: List of ToolSpec instances.

    Returns:
        List of instantiated tool objects.
    """
    tools = []
    for spec in tool_specs:
        try:
            if isinstance(spec, ClassToolSpec):
                tool_class = import_from_qualified(spec.tool_class)
                tools.append(tool_class(**spec.extra_params))
            elif isinstance(spec, FunctionToolSpec):
                tools.append(import_from_qualified(spec.function))
            elif isinstance(spec, FactoryToolSpec):
                result = import_from_qualified(spec.factory)(**spec.extra_params)
                tools.extend(result if isinstance(result, list) else [result])
        except ModuleNotFoundError as ex:
            missing_module = str(ex).split("'")[1] if "'" in str(ex) else str(ex)
            logger.warning(f"Skipping tool: missing optional dependency '{missing_module}'")
        except (ImportError, AttributeError) as ex:
            logger.warning(f"Skipping tool {spec}: {ex}")
        except Exception as ex:
            logger.warning(f"Failed to instantiate tool from {spec}: {ex}")
    return tools


def convert_langchain_tools(tools: list[Any]) -> list[Any]:
    """Convert LangChain BaseTool instances to SmolAgent Tools.

    Args:
        tools: List of tool instances that may include LangChain tools.

    Returns:
        List of tools with LangChain tools converted to SmolAgent tools.
    """
    return [SmolAgentTool.from_langchain(t) if isinstance(t, LangChainBaseTool) else t for t in tools]


def _default_smolagents_path() -> Path:
    agents_dir = paths_config().config / "agents"
    dir_path = agents_dir / "smolagents"
    return dir_path if dir_path.is_dir() else agents_dir / "smolagents.yaml"


def _build_smolagent_config(raw: dict[str, Any]) -> SmolagentsAgentConfig:
    """Build a ``SmolagentsAgentConfig`` from a raw YAML entry dict."""
    from pydantic import TypeAdapter

    ta = TypeAdapter(ToolSpec)
    tool_specs = []
    for tool_cfg in raw.get("tools", []):
        try:
            tool_specs.append(ta.validate_python(tool_cfg))
        except Exception:
            pass
    raw_tools = instantiate_tools_from_specs(tool_specs)
    return SmolagentsAgentConfig(
        name=raw.get("name", ""),
        tool_specs=tool_specs,
        tools=convert_langchain_tools(raw_tools),
        mcp_servers=raw.get("mcp_servers", []),
        examples=raw.get("examples", []),
        authorized_imports=raw.get("authorized_imports", []),
        pre_prompt=raw.get("pre_prompt"),
    )


def load_all_demos_from_config() -> list[SmolagentsAgentConfig]:
    """Load and configure all SmolAgent demonstration scenarios.

    Returns:
        List of configured ``SmolagentsAgentConfig`` objects.
    """
    try:
        entries: list[dict] = load_yaml_configs(_default_smolagents_path(), "smolagents_codeact")  # type: ignore[assignment]
        return [_build_smolagent_config(raw) for raw in entries]
    except Exception as ex:
        logger.exception(f"Failed to load demo configurations: {ex}")
        return []


def create_demo_from_config(config_name: str) -> SmolagentsAgentConfig | None:
    """Create a single ``SmolagentsAgentConfig`` from configuration.

    Args:
        config_name: Name of the configuration to load.

    Returns:
        ``SmolagentsAgentConfig`` instance or ``None`` if not found.
    """
    try:
        entries: list[dict] = load_yaml_configs(_default_smolagents_path(), "smolagents_codeact")  # type: ignore[assignment]
        for raw in entries:
            if raw.get("name", "").lower() == config_name.lower():
                return _build_smolagent_config(raw)
    except Exception as ex:
        logger.error(f"Failed to load demo config '{config_name}': {ex}")
    return None
