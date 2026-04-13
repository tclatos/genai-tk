"""Shared tool configuration loader for LangChain-based agents.

Provides ``process_langchain_tools_from_config`` which converts ``ToolSpec``
objects into ``BaseTool`` instances.  Agent-level config loading has moved to
``genai_tk.agents.langchain.config``.
"""

import inspect
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec, ToolSpec
from genai_tk.utils.config_mngr import import_from_qualified


def process_langchain_tools_from_config(tools_config: list[ToolSpec] | None, llm: Any = "default") -> list[BaseTool]:
    """Instantiate LangChain tools from a list of ``ToolSpec`` objects.

    Args:
        tools_config: Parsed tool specification models, or ``None``.
        llm: LLM instance or identifier passed to factory functions that accept it.

    Returns:
        List of LangChain BaseTool instances.
    """
    if not tools_config:
        return []

    tools: list[BaseTool] = []
    for spec in tools_config:
        try:
            if isinstance(spec, FunctionToolSpec):
                tools.extend(_process_function_tool(spec))
            elif isinstance(spec, ClassToolSpec):
                tool_instance = _process_class_tool(spec)
                if tool_instance:
                    tools.append(tool_instance)
            elif isinstance(spec, FactoryToolSpec):
                tools.extend(_process_factory_tool(spec, llm=llm))
        except Exception as ex:
            raise Exception(f"Failed to process tool {spec!r}: {ex}") from ex

    return tools


def _process_function_tool(spec: FunctionToolSpec) -> list[BaseTool]:
    tool_func = import_from_qualified(spec.function)
    if isinstance(tool_func, BaseTool):
        return [tool_func]
    if callable(tool_func):
        result = tool_func()
        if isinstance(result, BaseTool):
            return [result]
        if isinstance(result, list):
            return [t for t in result if isinstance(t, BaseTool)]
    return []


def _process_class_tool(spec: ClassToolSpec) -> BaseTool | None:
    try:
        tool_class = import_from_qualified(spec.tool_class)
        instance = tool_class(**spec.extra_params)
        if isinstance(instance, BaseTool):
            return instance
        logger.warning("Class {!r} does not produce a BaseTool instance", spec.tool_class)
    except ModuleNotFoundError as ex:
        missing_module = str(ex).split("'")[1] if "'" in str(ex) else str(ex)
        logger.warning("Skipping tool {!r}: missing optional dependency '{}'", spec.tool_class, missing_module)
    except (ImportError, AttributeError) as ex:
        logger.warning("Skipping tool {!r}: {}", spec.tool_class, ex)
    except Exception as ex:
        logger.warning("Failed to load class {!r}: {}", spec.tool_class, ex)
    return None


def _process_factory_tool(spec: FactoryToolSpec, llm: Any = "default") -> list[BaseTool]:
    params = dict(spec.extra_params)
    try:
        factory_func = import_from_qualified(spec.factory)
        if "llm" in inspect.signature(factory_func).parameters:
            params["llm"] = llm
        tool_result = factory_func(**params)
        if isinstance(tool_result, list):
            return [t for t in tool_result if isinstance(t, BaseTool)]
        if isinstance(tool_result, BaseTool):
            return [tool_result]
    except Exception as ex:
        logger.warning("Failed to load factory {!r}: {}", spec.factory, ex)
    return []
