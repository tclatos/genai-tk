"""Tests for process_langchain_tools_from_config in shared_config_loader.

The agent-level config loading previously in this module has moved to
``genai_tk.agents.langchain.config``.  These tests cover the remaining
public API: tool instantiation from raw YAML specifications.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from genai_tk.tools.langchain.shared_config_loader import process_langchain_tools_from_config

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

    def test_non_dict_items_are_skipped(self) -> None:
        result = process_langchain_tools_from_config(["string_item", 42])  # type: ignore[arg-type]
        assert result == []


# ---------------------------------------------------------------------------
# Function tools
# ---------------------------------------------------------------------------


class TestFunctionTools:
    def test_function_tool_returning_base_tool(self) -> None:
        fake = _make_tool("my_func_tool")
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=fake):
            result = process_langchain_tools_from_config([{"function": "some.mod:tool_func"}])
        assert result == [fake]

    def test_function_tool_callable_returning_tool(self) -> None:
        fake = _make_tool("returned_tool")

        def factory() -> BaseTool:
            return fake

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory):
            result = process_langchain_tools_from_config([{"function": "some.mod:factory"}])
        assert result == [fake]

    def test_function_tool_callable_returning_list(self) -> None:
        tools = [_make_tool("tool_a"), _make_tool("tool_b")]

        def factory():
            return tools

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory):
            result = process_langchain_tools_from_config([{"function": "some.mod:factory"}])
        assert result == tools

    def test_function_tool_invalid_ref_logs_warning(self) -> None:
        result = process_langchain_tools_from_config([{"function": "no_colon_here"}])
        assert result == []

    def test_function_tool_import_error_raises(self) -> None:
        with patch(
            "genai_tk.tools.langchain.shared_config_loader.import_from_qualified",
            side_effect=ImportError("boom"),
        ):
            with pytest.raises(Exception, match="boom"):
                process_langchain_tools_from_config([{"function": "bad.mod:tool"}])


# ---------------------------------------------------------------------------
# Class tools
# ---------------------------------------------------------------------------


class TestClassTools:
    def test_class_tool_instantiation(self) -> None:
        fake = _make_tool("class_tool")
        FakeCls = MagicMock(return_value=fake)
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=FakeCls):
            result = process_langchain_tools_from_config([{"class": "some.mod:FakeTool"}])
        assert result == [fake]

    def test_class_tool_invalid_ref_returns_empty(self) -> None:
        result = process_langchain_tools_from_config([{"class": "no_colon"}])
        assert result == []

    def test_class_tool_non_basetool_returns_empty(self) -> None:
        class NotATool:
            pass

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=NotATool):
            result = process_langchain_tools_from_config([{"class": "mod:NotATool"}])
        assert result == []


# ---------------------------------------------------------------------------
# Factory tools
# ---------------------------------------------------------------------------


class TestFactoryTools:
    def test_factory_tool_returns_list(self) -> None:
        tools = [_make_tool("factory_tool")]

        def factory_func():
            return tools

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            result = process_langchain_tools_from_config([{"factory": "mod:factory_func"}])
        assert result == tools

    def test_factory_tool_passes_llm_when_supported(self) -> None:
        received: dict = {}

        def factory_func(llm=None):
            received["llm"] = llm
            return []

        llm = MagicMock()
        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            process_langchain_tools_from_config([{"factory": "mod:factory_func"}], llm=llm)
        assert received.get("llm") is llm

    def test_factory_tool_does_not_pass_llm_when_not_accepted(self) -> None:
        def factory_func():
            return []

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", return_value=factory_func):
            result = process_langchain_tools_from_config([{"factory": "mod:factory_func"}], llm=MagicMock())
        assert result == []

    def test_factory_tool_invalid_ref_returns_empty(self) -> None:
        result = process_langchain_tools_from_config([{"factory": "no_colon"}])
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

        with patch("genai_tk.tools.langchain.shared_config_loader.import_from_qualified", side_effect=_import):
            result = process_langchain_tools_from_config(
                [
                    {"function": "mod:func_tool"},
                    {"class": "mod:ClassTool"},
                ]
            )
        assert func_tool in result
        assert class_tool in result
