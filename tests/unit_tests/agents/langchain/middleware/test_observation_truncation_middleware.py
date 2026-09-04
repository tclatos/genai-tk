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


def test_target_tools_filtering():
    mw = ObservationTruncationMiddleware(max_chars=20, tools=["get_section_content"])
    req_match = DummyRequest("get_section_content")
    req_other = DummyRequest("other_tool")
    long_text = "A" * 100

    assert "Truncated" in mw.wrap_tool_call(req_match, lambda r: long_text)
    assert mw.wrap_tool_call(req_other, lambda r: long_text) == long_text


def test_langchain_agent_factory_compatibility():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    mw = ObservationTruncationMiddleware(max_chars=100)
    # create_agent scans [t for m in middleware for t in getattr(m, "tools", [])]
    # Verify ObservationTruncationMiddleware does not expose a None tools attribute
    agent = create_agent(
        model=FakeListChatModel(responses=["test"]),
        tools=[],
        middleware=[mw],
    )
    assert agent is not None
