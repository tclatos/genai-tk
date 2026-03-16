"""Unit tests for genai_tk.agents.langchain.langchain_agent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from genai_tk.agents.langchain.langchain_agent import LangchainAgent, _extract_content

# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_langgraph_state_dict(self) -> None:
        msg = AIMessage(content="Hello world")
        result = _extract_content({"messages": [HumanMessage(content="hi"), msg]})
        assert result == "Hello world"

    def test_langgraph_state_empty_messages(self) -> None:
        result = _extract_content({"messages": []})
        assert result == ""

    def test_direct_message(self) -> None:
        msg = AIMessage(content="Response text")
        assert _extract_content(msg) == "Response text"

    def test_plain_string(self) -> None:
        assert _extract_content("raw string") == "raw string"

    def test_list_content(self) -> None:
        msg = AIMessage(content=["block one", "block two"])
        result = _extract_content(msg)
        assert result == "block one\nblock two"

    def test_dict_without_messages_key(self) -> None:
        # Falls back to str()
        result = _extract_content({"other": "data"})
        assert "other" in result

    def test_none_like_object(self) -> None:
        # An object with no content attr falls back to str()
        result = _extract_content(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# LangchainAgent – construction
# ---------------------------------------------------------------------------


MINIMAL_CONFIG: dict[str, Any] = {
    "langchain_agents": {
        "defaults": {"type": "react"},
        "default_profile": "simple",
        "profiles": [
            {
                "name": "simple",
                "type": "react",
                "llm": "parrot_local@fake",
                "tools": [],
                "mcp_servers": [],
            }
        ],
    }
}


def _make_config():
    """Return a minimal LangchainAgentsConfig for mocking."""
    from genai_tk.agents.langchain.config import (
        AgentDefaults,
        AgentProfileConfig,
        LangchainAgentsConfig,
    )

    profile = AgentProfileConfig(name="simple", type="react", llm="parrot_local@fake")
    return LangchainAgentsConfig(
        defaults=AgentDefaults(),
        default_profile="simple",
        profiles=[profile],
    )


class TestLangchainAgentConstruction:
    def test_adhoc_no_profile(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake")
        assert agent._profile is not None
        assert agent._profile.name == "adhoc"
        assert agent._profile.type == "react"
        assert agent._profile.llm == "parrot_local@fake"

    def test_adhoc_with_agent_type(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake", agent_type="deep")
        assert agent._profile.type == "deep"

    def test_adhoc_with_system_prompt(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake", system_prompt="Be concise")
        assert agent._profile.system_prompt == "Be concise"

    def test_profile_from_config(self) -> None:
        cfg = _make_config()
        with patch("genai_tk.agents.langchain.config.load_unified_config", return_value=cfg):
            agent = LangchainAgent("simple")
        assert agent._profile.name == "simple"
        assert agent._profile.llm == "parrot_local@fake"

    def test_profile_llm_override(self) -> None:
        cfg = _make_config()
        with patch("genai_tk.agents.langchain.config.load_unified_config", return_value=cfg):
            agent = LangchainAgent("simple", llm="other_llm@fake")
        assert agent._profile.llm == "other_llm@fake"

    def test_profile_system_prompt_override(self) -> None:
        cfg = _make_config()
        with patch("genai_tk.agents.langchain.config.load_unified_config", return_value=cfg):
            agent = LangchainAgent("simple", system_prompt="Custom prompt")
        assert agent._profile.system_prompt == "Custom prompt"

    def test_profile_mcp_servers_merged(self) -> None:
        cfg = _make_config()
        with patch("genai_tk.agents.langchain.config.load_unified_config", return_value=cfg):
            agent = LangchainAgent("simple", mcp_servers=["extra-server"])
        assert "extra-server" in agent._profile.mcp_servers

    def test_invalid_profile_raises(self) -> None:
        cfg = _make_config()
        with patch("genai_tk.agents.langchain.config.load_unified_config", return_value=cfg):
            with pytest.raises(ValueError, match="not found"):
                LangchainAgent("nonexistent")

    def test_agent_not_initialized_on_construction(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake")
        assert agent._agent is None

    def test_checkpointer_flag(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake", checkpointer=True)
        assert agent.checkpointer is True

    def test_details_flag(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake", details=True)
        assert agent.details is True


# ---------------------------------------------------------------------------
# LangchainAgent – lazy initialization
# ---------------------------------------------------------------------------


class TestLangchainAgentInit:
    @pytest.mark.asyncio
    async def test_ensure_initialized_creates_agent(self) -> None:
        mock_compiled = MagicMock()
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            agent = LangchainAgent(llm="parrot_local@fake")
            result = await agent._ensure_initialized()

        assert result is mock_compiled
        assert agent._agent is mock_compiled

    @pytest.mark.asyncio
    async def test_ensure_initialized_caches_agent(self) -> None:
        mock_compiled = MagicMock()
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ) as mock_factory:
            agent = LangchainAgent(llm="parrot_local@fake")
            await agent._ensure_initialized()
            await agent._ensure_initialized()

        assert mock_factory.call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_initialized_passes_checkpointer_flag(self) -> None:
        mock_compiled = MagicMock()
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ) as mock_factory:
            agent = LangchainAgent(llm="parrot_local@fake", checkpointer=True)
            await agent._ensure_initialized()

        _args, kwargs = mock_factory.call_args
        assert kwargs["force_memory_checkpointer"] is True


# ---------------------------------------------------------------------------
# LangchainAgent – arun / run
# ---------------------------------------------------------------------------


class TestLangchainAgentRun:
    @pytest.mark.asyncio
    async def test_arun_returns_string(self) -> None:
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="The answer is 42")]}

        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            agent = LangchainAgent(llm="parrot_local@fake")
            result = await agent.arun("What is the answer?")

        assert result == "The answer is 42"

    def test_run_sync_wrapper(self) -> None:
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="sync result")]}

        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            agent = LangchainAgent(llm="parrot_local@fake")
            result = agent.run("hello")

        assert result == "sync result"

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self) -> None:
        async def _mock_stream(messages: Any) -> Any:
            for chunk in [
                {"messages": [AIMessage(content="Hello")]},
                {"messages": [AIMessage(content=" world")]},
            ]:
                yield chunk

        mock_compiled = MagicMock()
        mock_compiled.astream = _mock_stream

        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            agent = LangchainAgent(llm="parrot_local@fake")
            chunks = []
            async for chunk in agent.astream("tell me"):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]


# ---------------------------------------------------------------------------
# LangchainAgent – close / context manager
# ---------------------------------------------------------------------------


class TestLangchainAgentClose:
    @pytest.mark.asyncio
    async def test_close_stops_backend(self) -> None:
        mock_backend = AsyncMock()
        mock_compiled = MagicMock()
        mock_compiled._backend = mock_backend

        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            agent = LangchainAgent(llm="parrot_local@fake")
            await agent._ensure_initialized()
            await agent.close()

        mock_backend.stop.assert_called_once()
        assert agent._agent is None

    @pytest.mark.asyncio
    async def test_close_noop_when_not_initialized(self) -> None:
        agent = LangchainAgent(llm="parrot_local@fake")
        # Should not raise
        await agent.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        mock_compiled = MagicMock()
        mock_compiled._backend = None

        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
            return_value=mock_compiled,
        ):
            async with LangchainAgent(llm="parrot_local@fake") as agent:
                await agent._ensure_initialized()
                assert agent._agent is not None
            assert agent._agent is None
