"""Shared tool configuration loader for LangChain-based agents.

Provides ``process_langchain_tools_from_config`` which converts ``ToolSpec``
objects into ``BaseTool`` instances. Agent-level config loading has moved to
``genai_tk.agents.langchain.config``.
"""

import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.agents.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec, ToolSpec, UnifiedToolSpec
from genai_tk.config_mgmt.import_utils import ImportResolver

import_from_qualified = ImportResolver.import_from_qualified


def process_langchain_tools_from_config(
    tools_config: list[ToolSpec | str | dict[str, Any] | BaseTool] | None, llm: Any = "default"
) -> list[BaseTool]:
    """Instantiate LangChain tools from a list of ``ToolSpec`` objects, qualified strings, dicts, or tool instances.

    Args:
        tools_config: Parsed tool specifications, qualified strings, dicts, or BaseTool instances.
        llm: LLM instance or identifier passed to factory functions that accept it.

    Returns:
        List of LangChain BaseTool instances.
    """
    if not tools_config:
        return []

    tools: list[BaseTool] = []
    for spec in tools_config:
        try:
            tools.extend(_process_tool_entry(spec, llm=llm))
        except Exception as ex:
            raise Exception(f"Failed to process tool {spec!r}: {ex}") from ex

    return tools


def _process_tool_entry(spec: Any, llm: Any = "default") -> list[BaseTool]:
    if isinstance(spec, BaseTool):
        return [spec]
    if callable(spec) and not inspect.isclass(spec):
        return _invoke_callable_tool(spec, {}, llm=llm)

    target: str | None = None
    params: dict[str, Any] = {}

    if isinstance(spec, str):
        target = spec
        params = {}
    elif isinstance(spec, dict):
        unified = UnifiedToolSpec.model_validate(spec)
        target = unified.target
        params = dict(unified.extra_params)
    elif isinstance(spec, UnifiedToolSpec):
        target = spec.target
        params = dict(spec.extra_params)
    elif isinstance(spec, ClassToolSpec):
        target = str(spec.tool_class)
        params = dict(spec.extra_params)
    elif isinstance(spec, FactoryToolSpec):
        target = str(spec.factory)
        params = dict(spec.extra_params)
    elif isinstance(spec, FunctionToolSpec):
        target = str(spec.function)
        params = {}
    elif hasattr(spec, "target"):
        target = str(spec.target)
        params = dict(getattr(spec, "extra_params", {}))

    if not target:
        return []

    return _load_tool_from_target(target, params, llm=llm)


def _load_tool_from_target(target: str, params: dict[str, Any], llm: Any = "default") -> list[BaseTool]:
    obj = import_from_qualified(target)

    if isinstance(obj, BaseTool):
        return [obj]

    if inspect.isclass(obj):
        instance = obj(**params)
        if isinstance(instance, BaseTool):
            return [instance]
        if isinstance(instance, list):
            return [t for t in instance if isinstance(t, BaseTool)]
        logger.warning("Class {!r} does not produce a BaseTool instance", target)
        return []

    if callable(obj):
        return _invoke_callable_tool(obj, params, llm=llm, target_name=target)

    return []


def _invoke_callable_tool(
    obj: Callable[..., Any], params: dict[str, Any], llm: Any = "default", target_name: str = ""
) -> list[BaseTool]:
    call_params = dict(params)
    try:
        sig = inspect.signature(obj)
        if "llm" in sig.parameters and "llm" not in call_params:
            call_params["llm"] = llm
        tool_result = obj(**call_params)
        if isinstance(tool_result, BaseTool):
            return [tool_result]
        if isinstance(tool_result, list):
            return [t for t in tool_result if isinstance(t, BaseTool)]
        if isinstance(tool_result, dict):
            return [t for t in tool_result.values() if isinstance(t, BaseTool)]
    except Exception as ex:
        logger.warning("Failed to invoke tool/factory {!r}: {}", target_name or obj, ex)
    return []
