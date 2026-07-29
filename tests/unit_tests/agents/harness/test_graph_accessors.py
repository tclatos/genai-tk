"""Unit tests for the ``BaseHarness.get_graph()`` / ``get_checkpointer()`` accessors.

Verifies:
- ``LangChainHarness.get_graph()`` returns the real compiled
  ``CompiledStateGraph`` for a react and a deep profile (no mocking of the
  factory — same style as ``test_factory.py``).
- ``DeerFlowHarness.get_graph()`` / ``get_checkpointer()`` delegate to
  ``EmbeddedDeerFlowClient.get_graph()`` / ``get_checkpointer()`` (mocked —
  a real DeerFlow config/agent build is out of scope for a unit test).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# LangChainHarness — real factory, fake LLM
# ---------------------------------------------------------------------------


async def test_langchain_harness_get_graph_react(fake_llm_id: str) -> None:
    from langgraph.graph.state import CompiledStateGraph

    from genai_tk.agents.harness.langchain_harness import LangChainHarness
    from genai_tk.agents.langchain.config import AgentProfileConfig

    profile = AgentProfileConfig(name="test-react", type="react", llm=fake_llm_id)
    harness = LangChainHarness(profile)

    graph = await harness.get_graph()

    assert isinstance(graph, CompiledStateGraph)


async def test_langchain_harness_get_graph_deep(fake_llm_id: str) -> None:
    from langgraph.graph.state import CompiledStateGraph

    from genai_tk.agents.harness.langchain_harness import LangChainHarness
    from genai_tk.agents.langchain.config import AgentProfileConfig

    profile = AgentProfileConfig(
        name="test-deep",
        type="deep",
        llm=fake_llm_id,
        enable_planning=False,
        enable_file_system=False,
    )
    harness = LangChainHarness(profile)

    graph = await harness.get_graph()

    assert isinstance(graph, CompiledStateGraph)


async def test_langchain_harness_get_graph_is_cached(fake_llm_id: str) -> None:
    """Calling get_graph() twice must not rebuild the agent."""
    from genai_tk.agents.harness.langchain_harness import LangChainHarness
    from genai_tk.agents.langchain.config import AgentProfileConfig

    profile = AgentProfileConfig(name="test-react", type="react", llm=fake_llm_id)
    harness = LangChainHarness(profile)

    graph1 = await harness.get_graph()
    graph2 = await harness.get_graph()

    assert graph1 is graph2


async def test_langchain_harness_get_checkpointer_none_by_default(fake_llm_id: str) -> None:
    from genai_tk.agents.harness.langchain_harness import LangChainHarness
    from genai_tk.agents.langchain.config import AgentProfileConfig, CheckpointerConfig

    profile = AgentProfileConfig(
        name="test-react", type="react", llm=fake_llm_id, checkpointer=CheckpointerConfig(type="none")
    )
    harness = LangChainHarness(profile)

    checkpointer = await harness.get_checkpointer()

    assert checkpointer is None


# ---------------------------------------------------------------------------
# DeerFlowHarness — mocked embedded client (no real DeerFlow config/build)
# ---------------------------------------------------------------------------


async def test_deerflow_harness_get_graph_delegates_to_client() -> None:
    from genai_tk.agents.deer_flow.profile import DeerFlowProfile
    from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness

    harness = DeerFlowHarness("test-deerflow-profile")
    fake_graph = MagicMock(name="compiled-graph")
    fake_client = MagicMock()
    fake_client.get_graph.return_value = fake_graph
    harness._client = fake_client
    harness._profile = DeerFlowProfile(name="test-deerflow-profile", subagent_enabled=True, plan_mode=False)
    harness._model_name = "gpt-fake"

    graph = await harness.get_graph()

    assert graph is fake_graph
    fake_client.get_graph.assert_called_once_with(
        model_name="gpt-fake",
        subagent_enabled=True,
        is_plan_mode=False,
    )


async def test_deerflow_harness_get_checkpointer_delegates_to_client() -> None:
    from genai_tk.agents.deer_flow.profile import DeerFlowProfile
    from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness

    harness = DeerFlowHarness("test-deerflow-profile")
    fake_checkpointer = MagicMock(name="checkpointer")
    fake_client = MagicMock()
    fake_client.get_checkpointer.return_value = fake_checkpointer
    harness._client = fake_client
    harness._profile = DeerFlowProfile(name="test-deerflow-profile")
    harness._model_name = "gpt-fake"

    checkpointer = await harness.get_checkpointer()

    assert checkpointer is fake_checkpointer


async def test_embedded_client_get_graph_calls_private_ensure_agent() -> None:
    """Guards the exact private-call shape documented in the contract test."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    wrapper = EmbeddedDeerFlowClient.__new__(EmbeddedDeerFlowClient)
    fake_upstream = MagicMock()
    fake_upstream._model_name = "default-model"
    fake_upstream._agent = "the-compiled-graph"
    wrapper._client = fake_upstream

    graph = wrapper.get_graph(subagent_enabled=True)

    assert graph == "the-compiled-graph"
    fake_upstream._ensure_agent.assert_called_once()
    (config,), _kwargs = fake_upstream._ensure_agent.call_args
    assert config["configurable"]["model_name"] == "default-model"
    assert config["configurable"]["subagent_enabled"] is True


def test_embedded_client_get_checkpointer_returns_owned_checkpointer() -> None:
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    wrapper = EmbeddedDeerFlowClient.__new__(EmbeddedDeerFlowClient)
    wrapper._checkpointer = "our-sqlite-saver"

    assert wrapper.get_checkpointer() == "our-sqlite-saver"
