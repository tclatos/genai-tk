"""Tests for process_langchain_tools_from_config in shared_config_loader.

The agent-level config loading previously in this module has moved to
``genai_tk.agents.langchain.config``.  These tests cover the remaining
public API: tool instantiation from typed ``ToolSpec`` objects.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from genai_tk.tools.langchain.shared_config_loader import process_langchain_tools_from_config
from genai_tk.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "fake_tool") -> BaseTool:
    """Return a minimal mock BaseTool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    return tool


# ---------------------------------------------------------------------------
# None / empty input
# ---------------------------------------------------------------------------


class TestProcessLangchainToolsEdgeCases:
    def test_none_returns_empty_list(self) -> None:
        assert process_langchain_tools_from_config(None) == []

    def test_empty_list(self) -> None:
        assert process_langchain_tools_from_config([]) == []


# ---------------------------------------------------------------------------
# Function tools
# ---------------------------------------------------------------------------


class TestFunctionTools:
    def test_function_tool_returning_base_tool(self) -> None:
        fake = _make_tool("my_func_tool")
        spec = FunctionToolSpec(function="some.mod:tool_func")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=fake):
            result = process_langchain_tools_from_config([spec])
        assert result == [fake]

    def test_function_tool_callable_returning_tool(self) -> None:
        fake = _make_tool("returned_tool")

        def factory() -> BaseTool:
            return fake

        spec = FunctionToolSpec(function="some.mod:factory")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory):
            result = process_langchain_tools_from_config([spec])
        assert result == [fake]

    def test_function_tool_callable_returning_list(self) -> None:
        tools = [_make_tool("tool_a"), _make_tool("tool_b")]

        def factory():
            return tools

        spec = FunctionToolSpec(function="some.mod:factory")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory):
            result = process_langchain_tools_from_config([spec])
        assert result == tools

    def test_function_tool_import_error_raises(self) -> None:
        spec = FunctionToolSpec(function="bad.mod:tool")
        with patch(
            "genai_tk.tools.langchain.shared_config_loader.import_from_qualified",
            side_effect=ImportError("boom"),
        ):
            with pytest.raises(Exception, match="boom"):
                process_langchain_tools_from_config([spec])


# ---------------------------------------------------------------------------
# Class tools
# ---------------------------------------------------------------------------


class TestClassTools:
    def test_class_tool_instantiation(self) -> None:
        fake = _make_tool("class_tool")
        FakeCls = MagicMock(return_value=fake)
        spec = ClassToolSpec.model_validate({"class": "some.mod:FakeTool"})
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=FakeCls):
            result = process_langchain_tools_from_config([spec])
        assert result == [fake]

    def test_class_tool_non_basetool_returns_empty(self) -> None:
        class NotATool:
            pass

        spec = ClassToolSpec.model_validate({"class": "mod:NotATool"})
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=NotATool):
            result = process_langchain_tools_from_config([spec])
        assert result == []


# ---------------------------------------------------------------------------
# Factory tools
# ---------------------------------------------------------------------------


class TestFactoryTools:
    def test_factory_tool_returns_list(self) -> None:
        tools = [_make_tool("factory_tool")]

        def factory_func():
            return tools

        spec = FactoryToolSpec(factory="mod:factory_func")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            result = process_langchain_tools_from_config([spec])
        assert result == tools

    def test_factory_tool_passes_llm_when_supported(self) -> None:
        received: dict = {}

        def factory_func(llm=None):
            received["llm"] = llm
            return []

        llm = MagicMock()
        spec = FactoryToolSpec(factory="mod:factory_func")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            process_langchain_tools_from_config([spec], llm=llm)
        assert received.get("llm") is llm

    def test_factory_tool_does_not_pass_llm_when_not_accepted(self) -> None:
        def factory_func():
            return []

        spec = FactoryToolSpec(factory="mod:factory_func")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            result = process_langchain_tools_from_config([spec], llm=MagicMock())
        assert result == []


# ---------------------------------------------------------------------------
# Mixed tool types
# ---------------------------------------------------------------------------


class TestMixedTools:
    def test_multiple_tool_configs(self) -> None:
        func_tool = _make_tool("func_tool")
        class_tool = _make_tool("class_tool")
        MockClass = MagicMock(return_value=class_tool)

        def _import(path: str):
            if "func" in path:
                return func_tool
            return MockClass

        specs = [
            FunctionToolSpec(function="mod:func_tool"),
            ClassToolSpec.model_validate({"class": "mod:ClassTool"}),
        ]
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", side_effect=_import):
            result = process_langchain_tools_from_config(specs)
        assert func_tool in result
        assert class_tool in result
