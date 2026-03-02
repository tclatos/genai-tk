"""Shared tool configuration loader for LangChain-based agents.

Provides ``process_langchain_tools_from_config`` which converts raw YAML tool
specs into ``BaseTool`` instances.  Agent-level config loading has moved to
``genai_tk.agents.langchain.config``.
"""

import inspect
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.utils.config_mngr import import_from_qualified


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
