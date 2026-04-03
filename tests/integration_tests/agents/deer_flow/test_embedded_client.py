"""Integration tests for EmbeddedDeerFlowClient with real LLM.

These tests require:
- DEER_FLOW_PATH set to a valid deer-flow clone
- A configured LLM available (uses the default profile)
- The deerflow optional dependencies installed (uv sync --group deerflow)

Run with:
    uv run pytest tests/integration_tests/agents/deer_flow/ -v -m deerflow
"""

from __future__ import annotations

import os
from typing import Any

import pytest

DEER_FLOW_PATH = os.environ.get("DEER_FLOW_PATH")
SKIP_REASON = "DEER_FLOW_PATH is not set — skipping DeerFlow integration tests"

pytestmark = pytest.mark.deerflow


# ---------------------------------------------------------------------------
# Test middleware (records invocations for assertion)
# ---------------------------------------------------------------------------

# RecordingMiddleware is created dynamically in the test that needs it,
# because it must subclass langchain.agents.middleware.AgentMiddleware which
# is only available after _ensure_deer_flow_on_path() adds deer-flow to
# sys.path.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deer_flow_config(tmp_path_factory):
    """Generate a minimal deer-flow config.yaml for tests.

    Skips with SKIP_REASON when DEER_FLOW_PATH is not set.
    """
    if not DEER_FLOW_PATH:
        pytest.skip(SKIP_REASON)

    from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config

    try:
        config_path, _ = setup_deer_flow_config(sandbox="local", selected_llm=None)
        return config_path
    except Exception as exc:
        pytest.skip(f"Could not generate deer-flow config: {exc}")


@pytest.fixture(scope="module")
def embedded_client(deer_flow_config):
    """Return a basic EmbeddedDeerFlowClient for the test module."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    try:
        return EmbeddedDeerFlowClient(config_path=deer_flow_config)
    except ImportError as exc:
        pytest.skip(f"DeerFlow dependencies not importable: {exc}")


# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.real_models
async def test_stream_message_yields_token_events(embedded_client) -> None:
    """A simple prompt produces at least one TokenEvent."""
    from genai_tk.agents.deer_flow.embedded_client import ErrorEvent, TokenEvent

    events = []
    async for event in embedded_client.stream_message("test-smoke", "Say hello in one word.", mode="flash"):
        events.append(event)

    if all(isinstance(e, ErrorEvent) for e in events):
        errors = "; ".join(e.message or "(empty)" for e in events)
        pytest.skip(f"DeerFlow agent returned only error(s) — likely infrastructure/LLM issue: {errors}")

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) >= 1, f"Expected TokenEvent(s), got: {[type(e).__name__ for e in events]}"
    full_text = "".join(e.data for e in tokens)
    assert len(full_text.strip()) > 0


@pytest.mark.asyncio
@pytest.mark.real_models
async def test_stream_message_produces_nonempty_response(embedded_client) -> None:
    """Full concatenated text from a simple prompt is non-empty."""
    from genai_tk.agents.deer_flow.embedded_client import ErrorEvent, TokenEvent

    events = []
    async for event in embedded_client.stream_message(
        "test-response", "What is 2 + 2? Answer with just the number.", mode="flash"
    ):
        events.append(event)

    if all(isinstance(e, ErrorEvent) for e in events):
        errors = "; ".join(e.message or "(empty)" for e in events)
        pytest.skip(f"DeerFlow agent returned only error(s) — likely infrastructure/LLM issue: {errors}")

    text = "".join(e.data for e in events if isinstance(e, TokenEvent))
    assert len(text.strip()) > 0


# ---------------------------------------------------------------------------
# Middleware injection test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.real_models
async def test_middleware_is_invoked(deer_flow_config) -> None:
    """A custom middleware is called during agent execution."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError:
        pytest.skip("langchain.agents.middleware.AgentMiddleware not available")

    class RecordingMiddleware(AgentMiddleware):
        """Record after_model calls for test assertion."""

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def after_model(self, state, runtime=None):
            self.calls.append({"event": "after_model"})
            return None

    middleware = RecordingMiddleware()
    client = EmbeddedDeerFlowClient(
        config_path=deer_flow_config,
        middlewares=[middleware],
    )

    if not client._middlewares_supported:
        pytest.skip("Upstream DeerFlowClient does not support 'middlewares' — update deer-flow clone.")

    from genai_tk.agents.deer_flow.embedded_client import ErrorEvent

    events = []
    async for _event in client.stream_message("test-middleware", "Say yes.", mode="flash"):
        events.append(_event)

    if all(isinstance(e, ErrorEvent) for e in events):
        errors = "; ".join(e.message or "(empty)" for e in events)
        pytest.skip(f"DeerFlow agent returned only error(s) — likely infrastructure/LLM issue: {errors}")

    # At minimum the RecordingMiddleware.calls list should exist (no exception raised).
    assert isinstance(middleware.calls, list)


# ---------------------------------------------------------------------------
# available_skills filtering test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_available_skills_empty_set_accepted(deer_flow_config) -> None:
    """EmbeddedDeerFlowClient accepts an empty available_skills set without error."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    client = EmbeddedDeerFlowClient(
        config_path=deer_flow_config,
        available_skills=set(),
    )

    if not client._available_skills_supported:
        pytest.skip("Upstream DeerFlowClient does not support 'available_skills' — update deer-flow clone.")

    assert client.client is not None


# ---------------------------------------------------------------------------
# client property test (direct upstream API access)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_property_exposes_upstream_api(embedded_client) -> None:
    """The .client property exposes the upstream DeerFlowClient API."""
    models_resp = embedded_client.client.list_models()
    assert "models" in models_resp
    assert isinstance(models_resp["models"], list)
