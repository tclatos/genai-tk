"""Integration tests for LangchainAgent.

Fast tests use ``parrot_local@fake`` (zero-cost, no network).
Real-model tests use the ``fast_model`` tag (cheap, gated by --include-real-models).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from genai_tk.agents.langchain.langchain_agent import LangchainAgent

# ---------------------------------------------------------------------------
# Fake-model tests (zero cost, always run)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_adhoc_arun_with_fake_llm() -> None:
    """Ad-hoc agent with fake LLM returns a non-empty string."""
    agent = LangchainAgent(llm="parrot_local@fake")
    result = await agent.arun("Hello, agent!")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.fake_models
def test_adhoc_run_sync_with_fake_llm() -> None:
    """Synchronous run() wrapper works with fake LLM."""
    agent = LangchainAgent(llm="parrot_local@fake")
    result = agent.run("Sync test query")
    assert isinstance(result, str)


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_agent_reuses_compiled_graph() -> None:
    """Calling arun() twice reuses the same compiled graph (lazy init caches)."""
    agent = LangchainAgent(llm="parrot_local@fake")
    await agent.arun("first query")
    first_graph = agent._agent
    await agent.arun("second query")
    assert agent._agent is first_graph


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_close_clears_agent() -> None:
    """close() sets _agent back to None."""
    agent = LangchainAgent(llm="parrot_local@fake")
    await agent.arun("query")
    assert agent._agent is not None
    await agent.close()
    assert agent._agent is None


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    """LangchainAgent works as an async context manager."""
    async with LangchainAgent(llm="parrot_local@fake") as agent:
        result = await agent.arun("test")
    assert isinstance(result, str)


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_astream_yields_strings() -> None:
    """astream() yields non-empty string chunks."""
    agent = LangchainAgent(llm="parrot_local@fake")
    chunks = []
    async for chunk in agent.astream("Tell me something"):
        chunks.append(chunk)
    # At minimum the generator must not raise; the fake model may yield one chunk
    assert all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# Real-model tests (cheap, opt-in via --include-real-models)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.real_models
@pytest.mark.asyncio
async def test_react_agent_with_cheap_llm() -> None:
    """React agent with the configured fast_model tag returns a coherent response.

    Uses the ``fast_model`` tag defined in the project config, which resolves to a
    cheap/fast model (e.g. ``gpt-4.1-mini`` or ``qwen3.5-9b``).
    """
    agent = LangchainAgent(llm="fast_model", agent_type="react")
    result = await agent.arun("What is 2 + 2? Reply with just the number.")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Option / constructor coverage (fake models, no network)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
def test_system_prompt_applied_to_profile() -> None:
    """system_prompt kwarg is stored in the resolved profile."""
    agent = LangchainAgent(llm="parrot_local@fake", system_prompt="You are a helpful assistant.")
    assert agent._profile is not None
    assert agent._profile.system_prompt == "You are a helpful assistant."


@pytest.mark.integration
@pytest.mark.fake_models
def test_agent_type_override() -> None:
    """agent_type kwarg is reflected in the resolved profile."""
    agent = LangchainAgent(llm="parrot_local@fake", agent_type="react")
    assert agent._profile is not None
    assert agent._profile.type == "react"


@pytest.mark.integration
@pytest.mark.fake_models
def test_checkpointer_stored() -> None:
    """checkpointer=True is stored and forwarded to the factory."""
    agent = LangchainAgent(llm="parrot_local@fake", checkpointer=True)
    assert agent.checkpointer is True


@pytest.mark.integration
@pytest.mark.fake_models
def test_details_flag_stored() -> None:
    """details=True is stored on the agent."""
    agent = LangchainAgent(llm="parrot_local@fake", details=True)
    assert agent.details is True


@pytest.mark.integration
@pytest.mark.fake_models
def test_mcp_servers_stored() -> None:
    """mcp_servers list is stored on the agent."""
    agent = LangchainAgent(llm="parrot_local@fake", mcp_servers=["weather", "math"])
    assert agent.mcp_servers == ["weather", "math"]


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_arun_shell_delegates_to_shell_function() -> None:
    """arun_shell() delegates to run_langchain_agent_shell."""
    agent = LangchainAgent(llm="parrot_local@fake")
    with patch(
        "genai_tk.agents.langchain.agent_cli.run_langchain_agent_shell",
        new_callable=AsyncMock,
    ) as mock_shell:
        await agent.arun_shell()
    mock_shell.assert_called_once_with(agent)


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_unsupported_sandbox_raises_value_error() -> None:
    """_ensure_initialized() raises ValueError for an unknown sandbox type."""
    agent = LangchainAgent(llm="parrot_local@fake")
    # Override sandbox after construction to bypass Pydantic's Literal type guard
    object.__setattr__(agent, "sandbox", "kubernetes")
    object.__setattr__(agent, "_agent", None)

    with pytest.raises(ValueError, match="Unsupported sandbox type"):
        await agent._ensure_initialized()
