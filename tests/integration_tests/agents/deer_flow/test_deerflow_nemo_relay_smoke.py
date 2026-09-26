"""Integration tests for DeerFlow + NeMo Relay ATOF trajectory capture.

Verifies that DeerFlow runs (via EmbeddedDeerFlowClient / DeerFlowHarness) emit
canonical ATOF scope/llm/tool/mark events to the NeMo Relay subscriber.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("nemo_relay")

from langchain_core.tools import tool

from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient
from genai_tk.agents.deer_flow.relay import NemoRelayDeerFlowMiddleware
from genai_tk.utils.nemo_relay_setup import (
    flush_nemo_relay_async,
    reset_nemo_relay,
    setup_nemo_relay,
)
from genai_tk.utils.tracing import reset_monitoring

pytestmark = [pytest.mark.deerflow, pytest.mark.integration]


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@pytest.fixture
def deer_flow_config():
    """Generate a minimal deer-flow config.yaml for tests."""
    from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config

    try:
        config_path, _, _ = setup_deer_flow_config(sandbox="local", selected_llm=None)
        return config_path
    except Exception as exc:
        pytest.skip(f"Could not generate deer-flow config: {exc}")


@pytest.mark.asyncio
async def test_deerflow_emits_nemo_relay_atof_events(deer_flow_config: Path, tmp_path: Path) -> None:
    """DeerFlow stream_message emits ATOF scope, llm, tool, and mark events."""
    reset_monitoring()
    reset_nemo_relay()
    atof_path = tmp_path / "events.jsonl"
    assert setup_nemo_relay(atof_path=atof_path), "nemo_relay subscriber did not activate"

    relay_mw = NemoRelayDeerFlowMiddleware(
        agent_name="deerflow_test",
        mode="flash",
        skills=["math"],
        sandbox="local",
    )

    client = EmbeddedDeerFlowClient(
        config_path=deer_flow_config,
        middlewares=[relay_mw],
        extra_tools=[add_numbers],
    )

    events_collected = []
    async for event in client.stream_message(
        "deerflow-relay-test-thread",
        "Use the add_numbers tool to calculate 17 + 25.",
        mode="flash",
    ):
        events_collected.append(event)

    await flush_nemo_relay_async()

    lines = [ln for ln in atof_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, f"No ATOF events written to {atof_path}"
    events = [json.loads(ln) for ln in lines]

    kinds = {ev.get("kind") for ev in events}
    assert "scope" in kinds, f"Expected scope events in ATOF stream, got kinds={kinds}"

    # Check for DeerFlow mark event
    marks = [ev for ev in events if ev.get("kind") == "mark"]
    deerflow_marks = [
        m for m in marks if m.get("metadata", {}).get("integration") == "deerflow"
    ]
    assert deerflow_marks, f"Expected at least one DeerFlow mark event, got marks={marks}"
    assert deerflow_marks[0]["data"]["harness"] == "deerflow"
    assert deerflow_marks[0]["data"]["agent_name"] == "deerflow_test"

    # Check for LLM and Tool scopes
    scope_cats = {
        ev.get("category") for ev in events if ev.get("kind") == "scope" and ev.get("scope_category") == "start"
    }
    assert "llm" in scope_cats, f"Expected an LLM scope, got categories={scope_cats}"
    assert "agent" in scope_cats, f"Expected an agent scope, got categories={scope_cats}"

    reset_nemo_relay()
    reset_monitoring()
