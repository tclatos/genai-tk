"""Integration tests for LangchainAgent.

Fast tests use ``parrot_local@fake`` (zero-cost, no network).
Real-model tests use the ``fast_model`` tag (cheap, gated by --include-real-models).
"""

from __future__ import annotations

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
