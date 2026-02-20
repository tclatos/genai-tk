"""Unit tests for DeerFlowClient (HTTP client for deer-flow)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.extra.agents.deer_flow.client import (
    DeerFlowClient,
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_lines(*events: tuple[str, str]) -> list[str]:
    """Build a list of SSE lines from (event_type, json_data) tuples."""
    lines = []
    for event_type, json_data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json_data}")
        lines.append("")  # blank line separator
    return lines


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_ok() -> None:
    """health_check returns True when server responds with 2xx."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client = DeerFlowClient()
        result = await client.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_connection_error() -> None:
    """health_check returns False on connection error."""
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client = DeerFlowClient()
        result = await client.health_check()

    assert result is False


# ---------------------------------------------------------------------------
# stream_run — SSE parsing
# ---------------------------------------------------------------------------


def _make_streaming_mock(lines: list[str]):
    """Build an async context manager that yields SSE lines via aiter_lines()."""
    mock_resp = AsyncMock()
    mock_resp.status_code = 200

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_resp.aiter_lines = _aiter_lines

    mock_client = AsyncMock()
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(return_value=mock_client)
    mock_outer.__aexit__ = AsyncMock(return_value=False)
    return mock_outer


@pytest.mark.asyncio
async def test_stream_run_yields_tokens() -> None:
    """stream_run yields TokenEvents for messages SSE events."""
    lines = _sse_lines(
        ("messages", '[{"type": "AIMessageChunk", "content": "Hello "}]'),
        ("messages", '[{"type": "AIMessageChunk", "content": "world"}]'),
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 2
    assert tokens[0].data == "Hello "
    assert tokens[1].data == "world"


@pytest.mark.asyncio
async def test_stream_run_yields_node_events() -> None:
    """stream_run yields NodeEvents for updates SSE events (excluding internal nodes)."""
    lines = _sse_lines(
        ("updates", '{"researcher": {"messages": []}}'),
        ("updates", '{"__start__": {}}'),  # should be filtered out
        ("updates", '{"reporter": {"messages": []}}'),
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    node_events = [e for e in events if isinstance(e, NodeEvent)]
    assert len(node_events) == 2
    assert node_events[0].node == "researcher"
    assert node_events[1].node == "reporter"


@pytest.mark.asyncio
async def test_stream_run_emits_node_events_per_update() -> None:
    """Each updates batch emits a NodeEvent even when the same node appears again (needed for tool-call tracking)."""
    lines = _sse_lines(
        ("updates", '{"researcher": {}}'),
        ("updates", '{"researcher": {}}'),  # same node again — both should be emitted
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    node_events = [e for e in events if isinstance(e, NodeEvent)]
    assert len(node_events) == 2


@pytest.mark.asyncio
async def test_stream_run_yields_error_event() -> None:
    """stream_run yields ErrorEvent on error SSE event."""
    lines = _sse_lines(
        ("error", '{"error": "something went wrong"}'),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert "something went wrong" in errors[0].message


@pytest.mark.asyncio
async def test_stream_run_http_error() -> None:
    """stream_run yields ErrorEvent when server returns HTTP 4xx/5xx."""
    mock_resp = AsyncMock()
    mock_resp.status_code = 500
    mock_resp.aread = AsyncMock(return_value=b"Internal Server Error")

    mock_client = AsyncMock()
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    mock_outer = AsyncMock()
    mock_outer.__aenter__ = AsyncMock(return_value=mock_client)
    mock_outer.__aexit__ = AsyncMock(return_value=False)

    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=mock_outer):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert "500" in errors[0].message


@pytest.mark.asyncio
async def test_stream_run_content_list_blocks() -> None:
    """stream_run also extracts text from content-list block format."""
    lines = _sse_lines(
        ("messages", '[{"type": "AIMessageChunk", "content": [{"type": "text", "text": "Block text"}]}]'),
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 1
    assert tokens[0].data == "Block text"


# ---------------------------------------------------------------------------
# create_thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_thread_returns_thread_id() -> None:
    """create_thread returns the thread_id from the response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"thread_id": "abc123", "metadata": {}})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client = DeerFlowClient()
        thread_id = await client.create_thread()

    assert thread_id == "abc123"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def test_url_construction() -> None:
    """Client constructs correct LangGraph and Gateway URLs."""
    client = DeerFlowClient(langgraph_url="http://myhost:2024", gateway_url="http://myhost:8001")
    assert client._lg("/threads") == "http://myhost:2024/threads"
    assert client._gw("/api/models") == "http://myhost:8001/api/models"


def test_url_strips_trailing_slash() -> None:
    """Trailing slash on base URL is stripped."""
    client = DeerFlowClient(langgraph_url="http://localhost:2024/", gateway_url="http://localhost:8001/")
    assert not client._lg_url.endswith("/")
    assert not client._gw_url.endswith("/")


# ---------------------------------------------------------------------------
# ToolCallEvent / ToolResultEvent extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_run_yields_tool_call_event() -> None:
    """Tool calls in AIMessage are yielded as ToolCallEvent."""
    import json

    ai_msg = json.dumps(
        {
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "web_search", "args": {"query": "test"}, "id": "tc1"}],
                }
            ]
        }
    )
    lines = _sse_lines(
        ("updates", '{"model": ' + ai_msg + "}"),
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "web_search"
    assert tool_calls[0].args == {"query": "test"}
    assert tool_calls[0].call_id == "tc1"


@pytest.mark.asyncio
async def test_stream_run_yields_tool_result_event() -> None:
    """ToolMessages in state_diff are yielded as ToolResultEvent."""
    import json

    tool_msg = json.dumps(
        {
            "messages": [
                {
                    "type": "tool",
                    "name": "web_search",
                    "content": "Search results here",
                    "tool_call_id": "tc1",
                }
            ]
        }
    )
    lines = _sse_lines(
        ("updates", '{"tools": ' + tool_msg + "}"),
        ("end", "{}"),
    )
    with patch("genai_tk.extra.agents.deer_flow.client.httpx.AsyncClient", return_value=_make_streaming_mock(lines)):
        client = DeerFlowClient()
        events = [e async for e in client.stream_run("thread1", "hi")]

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].tool_name == "web_search"
    assert results[0].content == "Search results here"
    assert results[0].call_id == "tc1"
