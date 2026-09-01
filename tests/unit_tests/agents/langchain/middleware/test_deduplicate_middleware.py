"""Unit tests for DeduplicateToolCallsMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from genai_tk.agents.langchain.middleware.deduplicate_middleware import DeduplicateToolCallsMiddleware


def _tool_request(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": name, "args": args})


def test_deduplicate_middleware_stub_mode() -> None:
    mw = DeduplicateToolCallsMiddleware(tools=["get_document_toc"], mode="stub")

    call_count = 0

    def mock_handler(req: Any) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result for {req.tool_call['name']}:{req.tool_call['args']}"

    req1 = _tool_request("get_document_toc", {"doc_id": "doc123"})
    res1 = mw.wrap_tool_call(req1, mock_handler)
    assert res1 == "Result for get_document_toc:{'doc_id': 'doc123'}"
    assert call_count == 1

    # Second identical call: should intercept and return stub notice without calling handler
    res2 = mw.wrap_tool_call(req1, mock_handler)
    assert "[Notice: Duplicate call to 'get_document_toc'" in res2
    assert "doc123" in res2
    assert call_count == 1  # handler was not called again

    # Different arguments: handler should be called
    req2 = _tool_request("get_document_toc", {"doc_id": "doc456"})
    res3 = mw.wrap_tool_call(req2, mock_handler)
    assert res3 == "Result for get_document_toc:{'doc_id': 'doc456'}"
    assert call_count == 2

    # Different tool name not in list: passthrough without intercept
    req_other = _tool_request("get_section_content", {"section_ids": "s1"})
    res4 = mw.wrap_tool_call(req_other, mock_handler)
    assert res4 == "Result for get_section_content:{'section_ids': 's1'}"
    assert call_count == 3


def test_deduplicate_middleware_cache_mode() -> None:
    mw = DeduplicateToolCallsMiddleware(tools=["get_document_toc"], mode="cache")

    call_count = 0

    def mock_handler(req: Any) -> str:
        nonlocal call_count
        call_count += 1
        return f"Full TOC payload #{call_count}"

    req = _tool_request("get_document_toc", {"doc_id": "doc123"})
    res1 = mw.wrap_tool_call(req, mock_handler)
    assert res1 == "Full TOC payload #1"
    assert call_count == 1

    # Second call in cache mode: returns cached output
    res2 = mw.wrap_tool_call(req, mock_handler)
    assert res2 == "Full TOC payload #1"
    assert call_count == 1


@pytest.mark.asyncio
async def test_deduplicate_middleware_async() -> None:
    mw = DeduplicateToolCallsMiddleware(tools=["get_folder_toc"], mode="stub")

    call_count = 0

    async def async_mock_handler(req: Any) -> str:
        nonlocal call_count
        call_count += 1
        return "Folder TOC"

    req = _tool_request("get_folder_toc", {"folder_id": None})
    res1 = await mw.awrap_tool_call(req, async_mock_handler)
    assert res1 == "Folder TOC"
    assert call_count == 1

    res2 = await mw.awrap_tool_call(req, async_mock_handler)
    assert "[Notice: Duplicate call to 'get_folder_toc'" in res2
    assert call_count == 1
