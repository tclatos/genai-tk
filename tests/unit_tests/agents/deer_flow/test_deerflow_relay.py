"""Unit tests for DeerFlow NeMo Relay integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from genai_tk.agents.deer_flow.relay import (
    NemoRelayDeerFlowMiddleware,
    add_deerflow_nemo_relay_integration,
    json_safe,
)

pytest.importorskip("nemo_relay")


def test_json_safe_primitives_and_collections() -> None:
    """json_safe serializes primitives, dicts, lists, and bytes safely."""
    assert json_safe(123) == 123
    assert json_safe("test") == "test"
    assert json_safe(True) is True
    assert json_safe(None) is None
    assert json_safe(b"bytes_data") == "<bytes: 10 bytes>"
    assert json_safe({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}


def test_add_deerflow_nemo_relay_integration() -> None:
    """add_deerflow_nemo_relay_integration appends NemoRelayDeerFlowMiddleware."""
    middlewares: list[Any] = []
    updated = add_deerflow_nemo_relay_integration(
        middlewares,
        agent_name="test_agent",
        mode="pro",
        skills=["skill1"],
        sandbox="local",
    )
    assert len(updated) == 1
    assert isinstance(updated[0], NemoRelayDeerFlowMiddleware)
    assert updated[0]._agent_name == "test_agent"
    assert updated[0]._mode == "pro"
    assert updated[0]._skills == ["skill1"]
    assert updated[0]._sandbox == "local"

    # Idempotent: repeated call does not add another instance
    again = add_deerflow_nemo_relay_integration(updated, agent_name="test_agent")
    assert len(again) == 1


def test_deerflow_relay_middleware_emits_configured_mark() -> None:
    """before_agent and abefore_agent emit the DeerFlow configured mark."""
    mw = NemoRelayDeerFlowMiddleware(
        agent_name="deerflow_agent",
        mode="flash",
        skills=["search"],
        sandbox="local",
    )

    with patch("nemo_relay.scope.event") as mock_event:
        mw.before_agent(state=MagicMock(), runtime=MagicMock())
        mock_event.assert_called_once()
        name, kwargs = mock_event.call_args[0][0], mock_event.call_args[1]
        assert name == "DeerFlow Configured"
        assert kwargs["data"]["harness"] == "deerflow"
        assert kwargs["data"]["agent_name"] == "deerflow_agent"
        assert kwargs["data"]["mode"] == "flash"
        assert kwargs["data"]["skills"] == ["search"]
        assert kwargs["metadata"]["integration"] == "deerflow"
        assert kwargs["metadata"]["deerflow_kind"] == "harness"
        assert kwargs["metadata"]["phase"] == "configured"
