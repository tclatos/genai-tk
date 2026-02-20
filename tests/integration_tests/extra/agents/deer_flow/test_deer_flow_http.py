"""Integration tests for the Deer-flow HTTP API.

These tests require a running Deer-flow instance.
They are skipped automatically when DEER_FLOW_PATH is not set or when the
servers are not reachable.

Run with:
    uv run pytest tests/integration_tests/extra/agents/deer_flow/ -v -m integration
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from genai_tk.extra.agents.deer_flow.client import DeerFlowClient, TokenEvent
from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

DEER_FLOW_PATH = os.getenv("DEER_FLOW_PATH", "")
SKIP_REASON = "DEER_FLOW_PATH is not set — skipping integration tests"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def running_servers():
    """Start deer-flow servers for the test module, stop on teardown."""
    if not DEER_FLOW_PATH:
        pytest.skip(SKIP_REASON)

    mgr = DeerFlowServerManager(deer_flow_path=DEER_FLOW_PATH, start_timeout=90.0)
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest_asyncio.fixture
async def client(running_servers: DeerFlowServerManager) -> DeerFlowClient:
    """Return a DeerFlowClient connected to the running servers."""
    return DeerFlowClient()


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_servers_are_running(running_servers: DeerFlowServerManager) -> None:
    """Both servers must be reachable after startup."""
    assert await running_servers.is_running()


@pytest.mark.asyncio
async def test_health_check(client: DeerFlowClient) -> None:
    """LangGraph /info endpoint responds."""
    assert await client.health_check()


# ---------------------------------------------------------------------------
# Thread lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_thread_returns_nonempty_id(client: DeerFlowClient) -> None:
    """create_thread returns a non-empty string ID."""
    thread_id = await client.create_thread()
    assert isinstance(thread_id, str)
    assert len(thread_id) > 0


# ---------------------------------------------------------------------------
# Streaming run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not DEER_FLOW_PATH, reason=SKIP_REASON)
async def test_stream_run_yields_at_least_one_token(client: DeerFlowClient) -> None:
    """A simple prompt should produce at least one TokenEvent."""
    thread_id = await client.create_thread()
    events = []
    async for event in client.stream_run(thread_id, "Say hello in one word."):
        events.append(event)

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) >= 1, f"Expected at least one TokenEvent, got events: {events}"


@pytest.mark.asyncio
@pytest.mark.skipif(not DEER_FLOW_PATH, reason=SKIP_REASON)
async def test_stream_run_full_response_nonempty(client: DeerFlowClient) -> None:
    """Full concatenated token response must be non-empty."""
    thread_id = await client.create_thread()
    text = "".join(e.data async for e in client.stream_run(thread_id, "Say hi.") if isinstance(e, TokenEvent))
    assert len(text.strip()) > 0
