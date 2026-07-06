"""Unit tests for ``rich_middleware.py``.

Exercises the RichToolCallMiddleware callback hooks (sync + async, compact +
detailed modes) with lightweight fake events, the ToolCallLimitMiddleware
limits, the SingleToolExecutorMiddleware single-tool flow, and the
``create_rich_agent_middlewares`` factory.  No real LLM is involved — the model
collaborator in SingleToolExecutorMiddleware is a tiny fake (an external LLM
boundary), and a real ``@tool`` function is used for tool execution.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from rich.console import Console

from genai_tk.agents.langchain.middleware.rich_middleware import (
    RichToolCallMiddleware,
    SingleToolExecutorMiddleware,
    ToolCallLimitMiddleware,
    create_rich_agent_middlewares,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _console() -> Console:
    """A Rich console writing to an in-memory buffer (keeps the terminal clean)."""
    return Console(file=io.StringIO(), force_terminal=False, width=80, record=True)


@tool
def echo(text: str) -> str:
    """Echo back the text."""
    return text


class _FakeBoundModel:
    """Fake bound model returned by ``bind_tools`` (LLM boundary stand-in)."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def ainvoke(self, _messages: Any) -> Any:
        return self._response


class _FakeModel:
    """Fake chat model exposing ``bind_tools`` (LLM boundary stand-in)."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def bind_tools(self, _tools: Any, tool_choice: Any = None) -> _FakeBoundModel:
        return _FakeBoundModel(self._response)


def _tool_request(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": name, "args": args})


# --------------------------------------------------------------------------- #
# Static helpers
# --------------------------------------------------------------------------- #


def test_extract_tool_metadata() -> None:
    name, args = RichToolCallMiddleware._extract_tool_metadata(_tool_request("search", {"q": "x"}))
    assert name == "search"
    assert args == {"q": "x"}


def test_extract_tool_metadata_missing() -> None:
    name, args = RichToolCallMiddleware._extract_tool_metadata(SimpleNamespace())
    assert name == "<unknown>"
    assert args == {}


def test_model_name_resolves_common_attrs() -> None:
    # _model_name reads request.model (an object) and probes its name attributes.
    assert RichToolCallMiddleware._model_name(SimpleNamespace(model=SimpleNamespace(model="gpt-x"))) == "gpt-x"
    assert (
        RichToolCallMiddleware._model_name(SimpleNamespace(model=SimpleNamespace(model_name="claude-y"))) == "claude-y"
    )
    assert (
        RichToolCallMiddleware._model_name(SimpleNamespace(model=SimpleNamespace(_model_name="gemini-z"))) == "gemini-z"
    )
    assert RichToolCallMiddleware._model_name(SimpleNamespace(model=SimpleNamespace(model_id="llama"))) == "llama"


def test_model_name_none_and_fallback() -> None:
    assert RichToolCallMiddleware._model_name(None) == "<unknown>"
    assert RichToolCallMiddleware._model_name(SimpleNamespace()) == "<unknown>"
    # No string-valued name attr on the model object → falls back to its type name.
    fallback = RichToolCallMiddleware._model_name(SimpleNamespace(model=SimpleNamespace(model=123)))
    assert fallback == "SimpleNamespace"


def test_response_content_str() -> None:
    assert RichToolCallMiddleware._response_content("hello") == "hello"


def test_response_content_list_blocks() -> None:
    content = RichToolCallMiddleware._response_content(SimpleNamespace(content=[{"text": "a"}, "b"]))
    assert "a" in content and "b" in content


def test_response_content_object() -> None:
    obj = SimpleNamespace(content="plain")
    assert RichToolCallMiddleware._response_content(obj) == "plain"


def test_summarize_result_returns_string() -> None:
    assert isinstance(RichToolCallMiddleware._summarize_result("sql_db_list_tables", "table1\ntable2"), str)


# --------------------------------------------------------------------------- #
# Compact mode helpers
# --------------------------------------------------------------------------- #


def test_compact_print_planning_marks_statuses() -> None:
    mw = RichToolCallMiddleware(console=_console())
    todos = [
        {"content": "step one", "status": "completed"},
        {"content": "step two", "status": "in_progress"},
        {"content": "step three", "status": "pending"},
    ]
    mw._compact_print_planning(todos)
    assert mw._planning_shown is True


def test_compact_print_planning_header_changes_on_update() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._compact_print_planning([{"content": "a", "status": "pending"}])
    first = mw._planning_shown
    mw._compact_print_planning([{"content": "b", "status": "pending"}])
    assert first is True  # still True; the second call is an "update"


def test_compact_print_step_increments_counter() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._compact_print_step("search", "found 3")
    mw._compact_print_step("read", "ok")
    assert mw._step_number == 2
    assert mw._execution_header_printed is True


def test_compact_print_skill_read_extracts_name() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._compact_print_skill_read("some/path/query-writing/SKILL.md")
    assert mw._step_number == 1


# --------------------------------------------------------------------------- #
# Detailed mode helpers
# --------------------------------------------------------------------------- #


def test_detail_print_tool_call_skill_vs_normal() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)
    # Should not raise for either branch (skill-read vs normal tool).
    mw._detail_print_tool_call("read_file", "x/SKILL.md")
    mw._detail_print_tool_call("search", {"q": "x"})


def test_detail_print_tool_result_truncates_long_content() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True, max_result_chars=10)
    long = "x" * 500
    mw._detail_print_tool_result("search", SimpleNamespace(content=long))
    # No assertion on output text; just that truncation path runs without error.


def test_detail_print_tool_result_markdown_and_plain() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)
    mw._detail_print_tool_result("search", SimpleNamespace(content="# Heading\n**bold**"))
    mw._detail_print_tool_result("search", SimpleNamespace(content="plain text"))


def test_detail_print_llm_call_with_skills_block() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)
    sys_msg = SimpleNamespace(content="**Available Skills:** **writing**: good  **How to Use Skills:** later")
    request = SimpleNamespace(
        model=SimpleNamespace(model="gpt-x"),
        messages=[HumanMessage(content="hello")],
        tools=[object()],
        system_message=sys_msg,
    )
    mw._call_count = 0
    mw._detail_print_llm_call(request)
    assert mw._call_count == 0  # only incremented in summary, not here


def test_detail_print_llm_call_no_system_message() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)
    request = SimpleNamespace(
        model=SimpleNamespace(model_name="claude"),
        messages=[HumanMessage(content="hi")],
        tools=[],
        system_message=None,
    )
    mw._detail_print_llm_call(request)  # should not raise


# --------------------------------------------------------------------------- #
# _handle_tool_call dispatch (compact mode)
# --------------------------------------------------------------------------- #


def test_handle_tool_call_write_todos_routes_to_planning() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._handle_tool_call("write_todos", {"todos": [{"content": "a", "status": "pending"}]}, SimpleNamespace(content=""))
    assert mw._planning_shown is True


def test_handle_tool_call_skill_read_routes_to_skill_read() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._handle_tool_call("read_file", {"path": "x/SKILL.md"}, SimpleNamespace(content="body"))
    assert mw._step_number == 1


def test_handle_tool_call_normal_routes_to_step() -> None:
    mw = RichToolCallMiddleware(console=_console())
    mw._handle_tool_call("search", {"q": "x"}, SimpleNamespace(content="result text"))
    assert mw._step_number == 1


# --------------------------------------------------------------------------- #
# wrap_tool_call / awrap_tool_call
# --------------------------------------------------------------------------- #


async def test_awrap_tool_call_compact_runs_handler() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        return SimpleNamespace(content="done")

    response = await mw.awrap_tool_call(_tool_request("search", {"q": "x"}), handler)
    assert response is not None
    assert mw._step_number == 1


async def test_awrap_tool_call_detailed_runs_handler() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)

    async def handler(_req: Any) -> Any:
        return SimpleNamespace(content="detailed result")

    await mw.awrap_tool_call(_tool_request("search", {"q": "x"}), handler)


def test_wrap_tool_call_sync() -> None:
    mw = RichToolCallMiddleware(console=_console())

    def handler(_req: Any) -> Any:
        return SimpleNamespace(content="sync result")

    response = mw.wrap_tool_call(_tool_request("search", {"q": "x"}), handler)
    assert response is not None
    assert mw._step_number == 1


# --------------------------------------------------------------------------- #
# wrap_model_call / awrap_model_call + _print_llm_response_summary
# --------------------------------------------------------------------------- #


async def test_awrap_model_call_text_response_increments_count() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        return AIMessage(content="hello world")

    await mw.awrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


async def test_awrap_model_call_tool_calls_response() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        return AIMessage(
            content="thinking", tool_calls=[{"name": "search", "args": {}, "id": "1", "type": "tool_call"}]
        )

    await mw.awrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


async def test_awrap_model_call_empty_response_warns() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        return AIMessage(content="")

    await mw.awrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


async def test_awrap_model_call_extended_model_response_unwrap() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        # ExtendedModelResponse → ModelResponse → result[0] is the AIMessage.
        return SimpleNamespace(model_response=SimpleNamespace(result=[AIMessage(content="unwrapped")]))

    await mw.awrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


async def test_awrap_model_call_detailed_empty_prints_debug_panel() -> None:
    mw = RichToolCallMiddleware(console=_console(), details=True)

    async def handler(_req: Any) -> Any:
        return AIMessage(content="")

    await mw.awrap_model_call(SimpleNamespace(model=SimpleNamespace(model="x"), messages=[], tools=[]), handler)
    assert mw._call_count == 1


async def test_awrap_model_call_handler_exception_reraised() -> None:
    mw = RichToolCallMiddleware(console=_console())

    async def handler(_req: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await mw.awrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


def test_wrap_model_call_sync_text() -> None:
    mw = RichToolCallMiddleware(console=_console())

    def handler(_req: Any) -> Any:
        return AIMessage(content="sync hello")

    mw.wrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


def test_wrap_model_call_sync_exception_reraised() -> None:
    mw = RichToolCallMiddleware(console=_console())

    def handler(_req: Any) -> Any:
        raise ValueError("sync fail")

    with pytest.raises(ValueError, match="sync fail"):
        mw.wrap_model_call(SimpleNamespace(), handler)
    assert mw._call_count == 1


# --------------------------------------------------------------------------- #
# create_rich_agent_middlewares
# --------------------------------------------------------------------------- #


def test_create_rich_agent_middlewares_default() -> None:
    mws = create_rich_agent_middlewares()
    assert len(mws) == 1
    assert isinstance(mws[0], RichToolCallMiddleware)
    assert mws[0]._details is False


def test_create_rich_agent_middlewares_details_flag() -> None:
    mws = create_rich_agent_middlewares(details=True)
    assert mws[0]._details is True


def test_create_rich_agent_middlewares_shares_console() -> None:
    shared = _console()
    mws = create_rich_agent_middlewares(console=shared)
    assert mws[0]._console is shared


# --------------------------------------------------------------------------- #
# ToolCallLimitMiddleware
# --------------------------------------------------------------------------- #


def test_tool_call_limit_sync_within_limit() -> None:
    mw = ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")

    def handler(req: Any) -> Any:
        return req

    assert mw.wrap_tool_call(SimpleNamespace(), handler) is not None
    assert mw.wrap_tool_call(SimpleNamespace(), handler) is not None
    assert mw._run_count == 2


def test_tool_call_limit_sync_error_when_exceeded() -> None:
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="error")

    def handler(req: Any) -> Any:
        return req

    mw.wrap_tool_call(SimpleNamespace(), handler)
    with pytest.raises(RuntimeError, match="Tool call limit exceeded"):
        mw.wrap_tool_call(SimpleNamespace(), handler)


def test_tool_call_limit_thread_limit_triggers_error() -> None:
    mw = ToolCallLimitMiddleware(run_limit=10, thread_limit=1, exit_behavior="error")

    def handler(req: Any) -> Any:
        return req

    mw.wrap_tool_call(SimpleNamespace(), handler)
    with pytest.raises(RuntimeError, match="Tool call limit exceeded"):
        mw.wrap_tool_call(SimpleNamespace(), handler)


async def test_tool_call_limit_async_within_limit() -> None:
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="end")

    async def handler(req: Any) -> Any:
        return req

    await mw.awrap_tool_call(SimpleNamespace(), handler)
    assert mw._run_count == 1


async def test_tool_call_limit_async_error_when_exceeded() -> None:
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="error")

    async def handler(req: Any) -> Any:
        return req

    await mw.awrap_tool_call(SimpleNamespace(), handler)
    with pytest.raises(RuntimeError, match="Tool call limit exceeded"):
        await mw.awrap_tool_call(SimpleNamespace(), handler)


# --------------------------------------------------------------------------- #
# SingleToolExecutorMiddleware
# --------------------------------------------------------------------------- #


async def test_single_tool_executor_no_tool_calls_returns_error_string() -> None:
    executor = SingleToolExecutorMiddleware()
    model = _FakeModel(SimpleNamespace(tool_calls=[]))

    result = await executor.execute_single_tool(echo, model, "ignored")
    assert result == "Error: Model did not generate a tool call"
    assert executor._executed is False


async def test_single_tool_executor_runs_tool_without_wrapper() -> None:
    executor = SingleToolExecutorMiddleware()
    model = _FakeModel(SimpleNamespace(tool_calls=[{"name": "echo", "args": {"text": "hi"}}]))

    result = await executor.execute_single_tool(echo, model, "anything")
    assert "hi" in result
    assert executor._executed is True


async def test_single_tool_executor_runs_tool_with_wrapper() -> None:
    executor = SingleToolExecutorMiddleware()
    model = _FakeModel(SimpleNamespace(tool_calls=[{"name": "echo", "args": {"text": "wrapped"}}]))
    wrapper = RichToolCallMiddleware(console=_console())

    # The executor calls tool_wrapper(request, handler) as an async callable;
    # pass the bound awrap_tool_call method (the documented "RichToolCallMiddleware" wrapper).
    result = await executor.execute_single_tool(echo, model, "anything", tool_wrapper=wrapper.awrap_tool_call)
    assert "wrapped" in result
    assert executor._executed is True
