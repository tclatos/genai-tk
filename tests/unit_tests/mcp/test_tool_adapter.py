"""Unit tests for genai_tk.mcp.tool_adapter – LangChain → FastMCP bridge."""

import asyncio
import inspect

from langchain_core.tools import BaseTool, tool
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from genai_tk.mcp.tool_adapter import (
    _make_mcp_wrapper,
    _params_from_pydantic,
    _safe_name,
    _str_result,
    register_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AddSchema(BaseModel):
    a: int
    b: int = 0


@tool
def add_numbers(a: int, b: int = 0) -> int:
    """Add two numbers."""
    return a + b


class _SimpleStringTool(BaseTool):
    name: str = "upper"
    description: str = "Convert text to upper-case."

    def _run(self, input: str) -> str:  # type: ignore[override]
        return input.upper()

    async def _arun(self, input: str) -> str:  # type: ignore[override]
        return input.upper()


class _StructuredTool(BaseTool):
    name: str = "adder"
    description: str = "Add two integers."
    args_schema: type[BaseModel] = _AddSchema

    def _run(self, a: int, b: int = 0) -> int:  # type: ignore[override]
        return a + b

    async def _arun(self, a: int, b: int = 0) -> int:  # type: ignore[override]
        return a + b


# ---------------------------------------------------------------------------
# _safe_name
# ---------------------------------------------------------------------------


class TestSafeName:
    def test_spaces_replaced(self) -> None:
        assert _safe_name("my tool") == "my_tool"

    def test_dashes_replaced(self) -> None:
        assert _safe_name("my-tool") == "my_tool"

    def test_dots_replaced(self) -> None:
        assert _safe_name("a.b.c") == "a_b_c"

    def test_already_safe(self) -> None:
        assert _safe_name("internet_search") == "internet_search"


# ---------------------------------------------------------------------------
# _str_result
# ---------------------------------------------------------------------------


class TestStrResult:
    def test_string_passthrough(self) -> None:
        assert _str_result("hello") == "hello"

    def test_dict_serialised(self) -> None:
        result = _str_result({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_other_converted(self) -> None:
        assert _str_result(42) == "42"


# ---------------------------------------------------------------------------
# _params_from_pydantic
# ---------------------------------------------------------------------------


class TestParamsFromPydantic:
    def test_field_count(self) -> None:
        params = _params_from_pydantic(_AddSchema)
        assert len(params) == 2

    def test_field_names(self) -> None:
        params = _params_from_pydantic(_AddSchema)
        names = [p.name for p in params]
        assert names == ["a", "b"]

    def test_required_has_no_default(self) -> None:
        params = _params_from_pydantic(_AddSchema)
        assert params[0].default is inspect.Parameter.empty

    def test_optional_has_default(self) -> None:
        params = _params_from_pydantic(_AddSchema)
        assert params[1].default == 0

    def test_annotations(self) -> None:
        params = _params_from_pydantic(_AddSchema)
        assert params[0].annotation is int
        assert params[1].annotation is int


# ---------------------------------------------------------------------------
# _make_mcp_wrapper – string-input tool
# ---------------------------------------------------------------------------


class TestMakeMcpWrapperSimple:
    def setup_method(self) -> None:
        self.lc_tool = _SimpleStringTool()
        self.wrapper = _make_mcp_wrapper(self.lc_tool)

    def test_name(self) -> None:
        assert self.wrapper.__name__ == "upper"

    def test_doc(self) -> None:
        assert "upper-case" in (self.wrapper.__doc__ or "")

    def test_is_async(self) -> None:
        assert asyncio.iscoroutinefunction(self.wrapper)

    def test_returns_string(self) -> None:
        result = asyncio.run(self.wrapper(input="hello"))
        assert result == "HELLO"


# ---------------------------------------------------------------------------
# _make_mcp_wrapper – structured tool
# ---------------------------------------------------------------------------


class TestMakeMcpWrapperStructured:
    def setup_method(self) -> None:
        self.lc_tool = _StructuredTool()
        self.wrapper = _make_mcp_wrapper(self.lc_tool)

    def test_name(self) -> None:
        assert self.wrapper.__name__ == "adder"

    def test_signature_has_params(self) -> None:
        sig = inspect.signature(self.wrapper)
        assert "a" in sig.parameters
        assert "b" in sig.parameters

    def test_signature_annotations(self) -> None:
        sig = inspect.signature(self.wrapper)
        assert sig.parameters["a"].annotation is int
        assert sig.parameters["b"].annotation is int

    def test_default_value(self) -> None:
        sig = inspect.signature(self.wrapper)
        assert sig.parameters["b"].default == 0

    def test_invocation(self) -> None:
        result = asyncio.run(self.wrapper(a=3, b=4))
        assert result == "7"


# ---------------------------------------------------------------------------
# register_tools – FastMCP integration
# ---------------------------------------------------------------------------


class TestRegisterTools:
    def test_tools_appear_in_server(self) -> None:
        server = FastMCP("test")
        register_tools(server, [_SimpleStringTool(), _StructuredTool()])
        tool_names = [t.name for t in server._tool_manager._tools.values()]
        assert "upper" in tool_names
        assert "adder" in tool_names

    def test_description_preserved(self) -> None:
        server = FastMCP("test")
        register_tools(server, [_SimpleStringTool()])
        tool_obj = list(server._tool_manager._tools.values())[0]
        assert "upper-case" in tool_obj.description

    def test_empty_list(self) -> None:
        server = FastMCP("test")
        register_tools(server, [])
        assert len(server._tool_manager._tools) == 0
