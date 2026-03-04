"""Shared tool configuration loader for LangChain-based agents.

Provides ``process_langchain_tools_from_config`` which converts raw YAML tool
specs into ``BaseTool`` instances.  Agent-level config loading has moved to
``genai_tk.agents.langchain.config``.
"""

import inspect
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from genai_tk.utils.config_mngr import get_raw_config, import_from_qualified


def process_langchain_tools_from_config(
    tools_config: list[dict[str, Any]] | None, llm: Any | None = None
) -> list[BaseTool]:
    """Process tools configuration and return list of LangChain tool instances.

    Args:
        tools_config: List of tool configuration dictionaries, or None.
        llm: Optional LLM instance passed to factory functions that support it.

    Returns:
        List of LangChain BaseTool instances.
    """
    tools: list[BaseTool] = []

    if tools_config is None:
        return tools

    for tool_config in tools_config:
        if not isinstance(tool_config, dict):
            continue

        try:
            if "function" in tool_config:
                tools.extend(_process_function_tool(tool_config))
            elif "class" in tool_config:
                tool_instance = _process_class_tool(tool_config)
                if tool_instance:
                    tools.append(tool_instance)
            elif "factory" in tool_config:
                tools.extend(_process_factory_tool(tool_config, llm=llm))
        except Exception as ex:
            raise Exception(f"Failed to process tool config {tool_config}: {ex}") from ex

    return tools


def _resolve_config_vars(config: Any) -> Any:
    """Resolve OmegaConf variables in configuration values.

    Recursively processes the configuration to resolve any OmegaConf interpolations
    (e.g., ${paths.project}) using OmegaConf's resolution mechanism merged with
    the global application configuration.

    Args:
        config: Configuration value (can be dict, list, DictConfig, or primitive)

    Returns:
        Configuration with resolved variables
    """
    if isinstance(config, dict):
        # Convert to DictConfig and merge with global config for proper variable resolution
        try:
            cfg_dict = OmegaConf.create(config)
            # Merge with global config so that ${paths.project} etc. can be resolved
            merged = OmegaConf.merge(get_raw_config(), cfg_dict)
            resolved = OmegaConf.to_container(merged, resolve=True)
            # Extract only the parts that were in the original config
            if isinstance(resolved, dict):
                return {k: resolved[k] for k in config.keys() if k in resolved}
            return config
        except Exception:
            # If resolution fails, return config as-is
            return config
    elif isinstance(config, list):
        return [_resolve_config_vars(item) for item in config]
    elif isinstance(config, DictConfig):
        # Resolve DictConfig values by merging with global config
        try:
            merged = OmegaConf.merge(get_raw_config(), config)
            resolved = OmegaConf.to_container(merged, resolve=True)
            return resolved if isinstance(resolved, dict) else config
        except Exception:
            return config
    else:
        # Return primitive values as-is
        return config


def _process_function_tool(tool_config: dict[str, Any]) -> list[BaseTool]:
    """Process a function-based tool configuration."""
    func_ref = tool_config.get("function")
    tools: list[BaseTool] = []

    if isinstance(func_ref, str) and ":" in func_ref:
        tool_func = import_from_qualified(func_ref)
        if isinstance(tool_func, BaseTool):
            tools.append(tool_func)
        elif callable(tool_func):
            result = tool_func()
            if isinstance(result, BaseTool):
                tools.append(result)
            elif isinstance(result, list):
                tools.extend([t for t in result if isinstance(t, BaseTool)])
    else:
        logger.warning(f"Unknown function reference: {func_ref!r}")

    return tools


def _process_class_tool(tool_config: dict[str, Any]) -> BaseTool | None:
    """Process a class-based tool configuration."""
    class_ref = tool_config.get("class")
    params = {k: v for k, v in tool_config.items() if k != "class"}

    if isinstance(class_ref, str) and ":" in class_ref:
        try:
            tool_class = import_from_qualified(class_ref)
            instance = tool_class(**params)
            if isinstance(instance, BaseTool):
                return instance
            logger.warning(f"Class {class_ref!r} does not produce a BaseTool instance")
        except ModuleNotFoundError as ex:
            # Extract missing module name from error message
            missing_module = str(ex).split("'")[1] if "'" in str(ex) else str(ex)
            logger.warning(f"Skipping tool {class_ref!r}: missing optional dependency '{missing_module}'")
        except (ImportError, AttributeError) as ex:
            logger.warning(f"Skipping tool {class_ref!r}: {ex}")
        except Exception as ex:
            logger.warning(f"Failed to load class {class_ref!r}: {ex}")
    else:
        logger.warning(f"Unknown tool class reference: {class_ref!r}")

    return None


def _process_factory_tool(tool_config: dict[str, Any], llm: Any | None = None) -> list[BaseTool]:
    """Process a factory-based tool configuration.

    Args:
        tool_config: Tool configuration dictionary.
        llm: Optional LLM instance to pass to factory functions that support it.
    """
    factory_ref = tool_config.get("factory")
    params = {k: v for k, v in tool_config.items() if k != "factory"}

    # Resolve OmegaConf variables (e.g., ${paths.project}) in parameters
    params = _resolve_config_vars(params)

    tools: list[BaseTool] = []

    if isinstance(factory_ref, str) and ":" in factory_ref:
        try:
            factory_func = import_from_qualified(factory_ref)
            sig = inspect.signature(factory_func)
            if llm is not None and "llm" in sig.parameters:
                params["llm"] = llm

            tool_result = factory_func(**params)

            if isinstance(tool_result, list):
                tools.extend([t for t in tool_result if isinstance(t, BaseTool)])
            elif isinstance(tool_result, BaseTool):
                tools.append(tool_result)
        except Exception as ex:
            logger.warning(f"Failed to load factory {factory_ref!r}: {ex}")
    else:
        logger.warning(f"Unknown factory reference: {factory_ref!r}")

    return tools
