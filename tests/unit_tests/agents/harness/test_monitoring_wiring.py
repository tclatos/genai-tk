"""Unit tests verifying monitoring/tracing is wired into both harnesses.

Checks that:
- ``LangChainHarness._ensure_agent()`` calls ``setup_monitoring()`` and
  ``astream()`` passes ``get_monitoring_callbacks()`` via the RunnableConfig.
- ``LangchainAgent._ensure_initialized()`` calls ``setup_monitoring()`` and
  ``arun()``/``astream()`` pass callbacks via the RunnableConfig.
- ``DeerFlowHarness`` delegates to ``prepare_profile`` which calls
  ``setup_monitoring()`` (the DeerFlow side is already wired).

These tests mock the heavy dependencies (create_langchain_agent, setup_monitoring)
so no LLM or Docker is required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


async def _async_iter(items: list) -> AsyncIterator:
    """Create an async iterator from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# LangChainHarness
# ---------------------------------------------------------------------------


@pytest.fixture
def langchain_profile():
    """A minimal LangChain profile for the harness."""
    from genai_tk.agents.langchain.config import AgentProfileConfig

    return AgentProfileConfig(name="TestProfile", type="react", llm="fake_llm")


async def test_langchain_harness_ensure_agent_calls_setup_monitoring(langchain_profile) -> None:
    """LangChainHarness._ensure_agent() must call setup_monitoring() before creating the agent."""
    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    harness = LangChainHarness(langchain_profile)

    with (
        patch("genai_tk.agents.harness.langchain_harness.setup_monitoring") as mock_setup,
        patch("genai_tk.agents.harness.langchain_harness.apply_harness_trace_metadata") as mock_apply,
        patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
    ):
        await harness._ensure_agent()

    mock_setup.assert_called_once()
    mock_apply.assert_called_once()


async def test_langchain_harness_astream_passes_callbacks(langchain_profile) -> None:
    """LangChainHarness.astream() must include monitoring callbacks in the config."""
    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    fake_callback = MagicMock()
    fake_agent = MagicMock()
    fake_agent.astream_events = MagicMock(return_value=_async_iter([]))

    harness = LangChainHarness(langchain_profile)
    harness._agent = fake_agent

    with patch(
        "genai_tk.agents.harness.langchain_harness.get_monitoring_callbacks",
        return_value=[fake_callback],
    ):
        async for _ in harness.astream("hello"):
            pass

    fake_agent.astream_events.assert_called_once()
    config = fake_agent.astream_events.call_args.kwargs["config"]
    assert "callbacks" in config
    assert fake_callback in config["callbacks"]


async def test_langchain_harness_astream_no_callbacks_when_none_active(langchain_profile) -> None:
    """When get_monitoring_callbacks() returns [], no callbacks key is added."""
    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    fake_agent = MagicMock()
    fake_agent.astream_events = MagicMock(return_value=_async_iter([]))

    harness = LangChainHarness(langchain_profile)
    harness._agent = fake_agent

    with patch(
        "genai_tk.agents.harness.langchain_harness.get_monitoring_callbacks",
        return_value=[],
    ):
        async for _ in harness.astream("hello"):
            pass

    config = fake_agent.astream_events.call_args.kwargs["config"]
    assert "callbacks" not in config


# ---------------------------------------------------------------------------
# LangchainAgent
# ---------------------------------------------------------------------------


async def test_langchain_agent_ensure_initialized_calls_setup_monitoring(fake_llm_id) -> None:
    """LangchainAgent._ensure_initialized() must call setup_monitoring()."""
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    agent = LangchainAgent(llm=fake_llm_id)

    with (
        patch("genai_tk.utils.tracing.setup_monitoring") as mock_setup,
        patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
    ):
        await agent._ensure_initialized()

    mock_setup.assert_called_once()


async def test_langchain_agent_arun_passes_callbacks(fake_llm_id) -> None:
    """LangchainAgent.arun() must pass monitoring callbacks in the invoke config."""
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    fake_callback = MagicMock()
    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": []})

    agent = LangchainAgent(llm=fake_llm_id)
    agent._agent = fake_agent

    with patch(
        "genai_tk.utils.tracing.get_monitoring_callbacks",
        return_value=[fake_callback],
    ):
        await agent.arun("test query")

    fake_agent.ainvoke.assert_awaited_once()
    config = fake_agent.ainvoke.call_args.args[1]
    assert "callbacks" in config
    assert fake_callback in config["callbacks"]


async def test_langchain_agent_astream_passes_callbacks(fake_llm_id) -> None:
    """LangchainAgent.astream() must pass monitoring callbacks in the stream config."""
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    fake_callback = MagicMock()
    fake_agent = MagicMock()
    fake_agent.astream = MagicMock(return_value=_async_iter([]))

    agent = LangchainAgent(llm=fake_llm_id)
    agent._agent = fake_agent

    with patch(
        "genai_tk.utils.tracing.get_monitoring_callbacks",
        return_value=[fake_callback],
    ):
        async for _ in agent.astream("test query"):
            pass

    fake_agent.astream.assert_called_once()
    config = fake_agent.astream.call_args.kwargs["config"]
    assert "callbacks" in config
    assert fake_callback in config["callbacks"]


def test_langchain_agent_invoke_config_includes_callbacks() -> None:
    """LangchainAgent._invoke_config() includes callbacks when active."""
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    fake_callback = MagicMock()
    with patch(
        "genai_tk.utils.tracing.get_monitoring_callbacks",
        return_value=[fake_callback],
    ):
        config = LangchainAgent._invoke_config()

    assert config["configurable"]["thread_id"] == "1"
    assert "callbacks" in config
    assert fake_callback in config["callbacks"]


def test_langchain_agent_invoke_config_no_callbacks_when_empty() -> None:
    """LangchainAgent._invoke_config() omits callbacks key when none active."""
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    with patch(
        "genai_tk.utils.tracing.get_monitoring_callbacks",
        return_value=[],
    ):
        config = LangchainAgent._invoke_config()

    assert "callbacks" not in config


# ---------------------------------------------------------------------------
# DeerFlow harness — setup_monitoring is delegated to prepare_profile
# ---------------------------------------------------------------------------


async def test_deerflow_harness_prepare_profile_calls_setup_monitoring() -> None:
    """DeerFlowHarness._ensure_client() delegates to prepare_profile which calls setup_monitoring()."""
    from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness

    harness = DeerFlowHarness("Research Assistant")

    fake_profile = MagicMock()
    fake_profile.name = "Research Assistant"
    fake_profile.mode = "pro"
    fake_profile.middlewares = []
    fake_profile.available_skills = None
    fake_profile.subagent_enabled = False
    fake_profile.plan_mode = True

    fake_client = MagicMock()

    with (
        patch(
            "genai_tk.agents.deer_flow.runtime.prepare_profile",
            new_callable=AsyncMock,
            return_value=(fake_profile, "model-id", MagicMock(), MagicMock()),
        ) as mock_prepare,
        patch(
            "genai_tk.agents.deer_flow.runtime.build_cli_middlewares",
            return_value=[],
        ),
        patch(
            "genai_tk.agents.deer_flow.embedded_client.EmbeddedDeerFlowClient",
            return_value=fake_client,
        ),
        patch("genai_tk.utils.tracing.apply_harness_trace_metadata") as mock_apply,
    ):
        await harness._ensure_client()

    mock_prepare.assert_awaited_once()
    mock_apply.assert_called_once()
