"""Unit tests for ObservationTruncationMiddleware."""

import pytest
from genai_tk.agents.langchain.middleware.observation_truncation_middleware import (
    ObservationTruncationMiddleware,
)


class DummyRequest:
    def __init__(self, name: str, args: dict | None = None):
        self.tool_call = {"name": name, "args": args or {}}


def test_no_truncation_below_limit():
    mw = ObservationTruncationMiddleware(max_chars=100)
    req = DummyRequest("get_section_content")
    result = mw.wrap_tool_call(req, lambda r: "Short content")
    assert result == "Short content"


def test_truncation_exceeds_max_chars():
    mw = ObservationTruncationMiddleware(max_chars=50, head_ratio=0.6)
    req = DummyRequest("get_section_content")
    long_text = "A" * 30 + "B" * 40 + "C" * 30
    result = mw.wrap_tool_call(req, lambda r: long_text)
    assert len(result) > 50  # notice added
    assert "Truncated" in result
    assert result.startswith("A" * 30)
    assert result.endswith("C" * 20)


@pytest.mark.asyncio
async def test_async_truncation():
    mw = ObservationTruncationMiddleware(max_chars=50, head_ratio=0.5)
    req = DummyRequest("get_section_content")
    long_text = "HEAD_" * 10 + "TAIL_" * 10

    async def async_handler(r):
        return long_text

    result = await mw.awrap_tool_call(req, async_handler)
    assert "Truncated" in result
    assert result.startswith("HEAD_")
    assert result.endswith("TAIL_")


def test_excluded_tools():
    mw = ObservationTruncationMiddleware(max_chars=20, excluded_tools=["web_search"])
    req = DummyRequest("web_search")
    long_text = "A" * 100
    result = mw.wrap_tool_call(req, lambda r: long_text)
    assert result == long_text


def test_line_truncation():
    mw = ObservationTruncationMiddleware(max_chars=10_000, max_lines=10, head_ratio=0.5)
    req = DummyRequest("get_section_content")
    lines = [f"Line {i}\n" for i in range(50)]
    text = "".join(lines)
    result = mw.wrap_tool_call(req, lambda r: text)
    assert "Truncated 40 lines" in result
    assert result.startswith("Line 0\n")
    assert result.endswith("Line 49\n")
