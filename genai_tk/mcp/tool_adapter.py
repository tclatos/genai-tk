"""Adapter layer: LangChain BaseTool → FastMCP tool registration.

For every LangChain ``BaseTool`` the adapter creates an async wrapper whose
Python signature mirrors the tool's Pydantic ``args_schema``.  FastMCP
inspects that signature (via ``inspect.signature``) to derive the JSON schema
it exposes to MCP clients.

The function body itself accepts ``**kwargs`` – FastMCP always calls tools with
keyword arguments, so the real signature only needs to be present as the
``__signature__`` attribute (which ``inspect.signature`` honours).

Example:
    ```python
    from mcp.server.fastmcp import FastMCP
    from genai_tk.mcp.tool_adapter import register_tools
    from genai_tk.tools.langchain.search_tools_factory import create_search_function

    server = FastMCP("demo")
    lc_tool = create_search_function()
    register_tools(server, [lc_tool])
    ```
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def register_tools(server: FastMCP, tools: list[BaseTool]) -> None:
    """Register a list of LangChain tools on a FastMCP server instance.

    Args:
        server: The FastMCP server to register tools on.
        tools: LangChain BaseTool instances to expose.
    """
    for tool in tools:
        wrapper = _make_mcp_wrapper(tool)
        server.add_tool(wrapper, name=_safe_name(tool.name), description=tool.description)
        logger.debug(f"Registered MCP tool: {tool.name!r}")


def resolve_tools_from_config(tool_configs: list) -> list[BaseTool]:
    """Resolve a list of tool config dicts (same format as langchain.yaml) into BaseTool instances.

    Args:
        tool_configs: List of tool configuration dicts, e.g.
            ``[{"factory": "module:func", "config": {...}}]``.

    Returns:
        Flat list of BaseTool instances.

    Example:
        ```python
        tools = resolve_tools_from_config([
            {"factory": "genai_tk.tools.langchain.search_tools_factory:create_search_function"}
        ])
        ```
    """
    from genai_tk.tools.langchain.shared_config_loader import process_langchain_tools_from_config

    return process_langchain_tools_from_config(tool_configs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Convert a tool name to a valid Python identifier for FastMCP."""
    return name.replace("-", "_").replace(" ", "_").replace(".", "_")


def _make_mcp_wrapper(lc_tool: BaseTool) -> Callable[..., Any]:
    """Build an async wrapper function whose signature FastMCP can introspect.

    The returned function accepts ``**kwargs`` at runtime but has a custom
    ``__signature__`` that lists individual parameters derived from the tool's
    ``args_schema``.  This satisfies both FastMCP schema generation and correct
    argument passing.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        Async callable with ``__name__``, ``__doc__``, and ``__signature__`` set.
    """
    schema: type[BaseModel] | None = lc_tool.args_schema
    tool_ref = lc_tool  # prevent closure mutation in loops

    if schema is not None and issubclass(schema, BaseModel):
        params = _params_from_pydantic(schema)

        async def wrapper(**kwargs: Any) -> str:
            return _str_result(await tool_ref.ainvoke(kwargs))

        wrapper.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
    else:
        # Fallback: single-string input.  FastMCP will call ``wrapper(input=x)``.
        # Use an explicit __signature__ so FastMCP sees the real ``str`` class
        # (not a PEP-563 lazy-annotation string which breaks ``issubclass``).

        async def wrapper(**kwargs: Any) -> str:  # type: ignore[misc]
            # Extract the single value and forward as a plain string to ainvoke
            value = next(iter(kwargs.values()), "") if kwargs else ""
            return _str_result(await tool_ref.ainvoke(str(value)))

        _str_param = inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
        wrapper.__signature__ = inspect.Signature([_str_param], return_annotation=str)  # type: ignore[attr-defined]

    wrapper.__name__ = _safe_name(lc_tool.name)
    wrapper.__doc__ = lc_tool.description
    return wrapper


def _params_from_pydantic(model: type[BaseModel]) -> list[inspect.Parameter]:
    """Build inspect.Parameter list from a Pydantic v2 model's fields.

    Args:
        model: A Pydantic BaseModel subclass.

    Returns:
        Ordered list of inspect.Parameter with correct annotations and defaults.
    """
    params: list[inspect.Parameter] = []
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation if field_info.annotation is not None else Any
        raw_default = field_info.default
        default = (
            inspect.Parameter.empty
            if isinstance(raw_default, type(PydanticUndefined)) or raw_default is PydanticUndefined
            else raw_default
        )
        params.append(
            inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
    return params


def _str_result(result: Any) -> str:
    """Convert any tool output to a string for MCP responses.

    Args:
        result: Raw tool output.

    Returns:
        String representation.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        import json

        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)
