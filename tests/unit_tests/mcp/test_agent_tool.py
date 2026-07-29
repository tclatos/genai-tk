"""Unit tests for the MCP agent-as-a-tool wrapper.

Verifies:
- Each call gets an isolated ``thread_id`` (fresh UUID) unless the caller
  explicitly passes one back to continue a conversation — no more shared
  ``"mcp_default"`` literal causing cross-session state bleed.
- The harness is built via ``create_harness`` (profile path) or a minimal
  adhoc ``LangChainHarness`` (no-profile path), and cached across calls.
- The tool returns a structured :class:`AgentToolResult`, and run failures
  are captured as a structured error rather than raised.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import tool
from mcp.server.fastmcp import FastMCP

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.mcp.agent_tool import AgentToolResult, register_agent_tool
from genai_tk.mcp.config import MCPAgentConfig

pytestmark = pytest.mark.unit


@tool
def _fake_tool(text: str) -> str:
    """A fake tool used only to exercise extra_tools plumbing."""
    return text


def _fake_server():
    """A minimal FastMCP stand-in that captures the registered tool function."""
    server = MagicMock(spec=FastMCP)
    captured: dict[str, object] = {}

    def _add_tool(fn, *, name, description):
        captured["fn"] = fn

    server.add_tool.side_effect = _add_tool
    return server, captured


async def test_invoke_agent_uses_fresh_isolated_thread_id_per_call() -> None:
    fake_harness = MagicMock(spec=BaseHarness)
    fake_harness.arun = AsyncMock(return_value="answer")

    cfg = MCPAgentConfig(name="run_test", profile="test-profile")
    server, captured = _fake_server()

    with patch("genai_tk.mcp.agent_tool._build_harness", new=AsyncMock(return_value=fake_harness)):
        register_agent_tool(server, cfg)
        invoke = captured["fn"]

        result1 = await invoke("hello")
        result2 = await invoke("hello again")

    assert isinstance(result1, AgentToolResult)
    assert result1.thread_id != result2.thread_id
    assert result1.thread_id.startswith("mcp-")
    thread_ids_used = [call.kwargs["thread_id"] for call in fake_harness.arun.await_args_list]
    assert thread_ids_used[0] != thread_ids_used[1]


async def test_invoke_agent_reuses_explicit_thread_id() -> None:
    fake_harness = MagicMock(spec=BaseHarness)
    fake_harness.arun = AsyncMock(return_value="answer")

    cfg = MCPAgentConfig(name="run_test", profile="test-profile")
    server, captured = _fake_server()

    with patch("genai_tk.mcp.agent_tool._build_harness", new=AsyncMock(return_value=fake_harness)):
        register_agent_tool(server, cfg)
        invoke = captured["fn"]

        result = await invoke("continue our chat", thread_id="conv-42")

    assert result.thread_id == "conv-42"
    fake_harness.arun.assert_awaited_once_with("continue our chat", thread_id="conv-42")


async def test_invoke_agent_returns_structured_text_result() -> None:
    fake_harness = MagicMock(spec=BaseHarness)
    fake_harness.arun = AsyncMock(return_value="42 is the answer")

    cfg = MCPAgentConfig(name="run_test", profile="test-profile")
    server, captured = _fake_server()

    with patch("genai_tk.mcp.agent_tool._build_harness", new=AsyncMock(return_value=fake_harness)):
        register_agent_tool(server, cfg)
        result = await captured["fn"]("what is the answer?")

    assert result.text == "42 is the answer"
    assert result.error is None


async def test_invoke_agent_captures_run_failure_as_structured_error() -> None:
    fake_harness = MagicMock(spec=BaseHarness)
    fake_harness.arun = AsyncMock(side_effect=RuntimeError("boom"))

    cfg = MCPAgentConfig(name="run_test", profile="test-profile")
    server, captured = _fake_server()

    with patch("genai_tk.mcp.agent_tool._build_harness", new=AsyncMock(return_value=fake_harness)):
        register_agent_tool(server, cfg)
        result = await captured["fn"]("trigger failure")

    assert result.error == "boom"
    assert result.text == ""


async def test_invoke_agent_caches_harness_across_calls() -> None:
    fake_harness = MagicMock(spec=BaseHarness)
    fake_harness.arun = AsyncMock(return_value="ok")
    build_mock = AsyncMock(return_value=fake_harness)

    cfg = MCPAgentConfig(name="run_test", profile="test-profile")
    server, captured = _fake_server()

    with patch("genai_tk.mcp.agent_tool._build_harness", new=build_mock):
        register_agent_tool(server, cfg)
        invoke = captured["fn"]
        await invoke("first")
        await invoke("second")

    build_mock.assert_awaited_once()


async def test_build_harness_with_profile_delegates_to_create_harness() -> None:
    from genai_tk.mcp.agent_tool import _build_harness

    fake_harness = MagicMock(spec=BaseHarness)
    cfg = MCPAgentConfig(name="run_test", profile="Research", llm="gpt_41mini@openai")
    extra_tools = [_fake_tool]

    with patch("genai_tk.agents.harness.registry.create_harness", return_value=fake_harness) as mock_create:
        result = await _build_harness(cfg, extra_tools)

    assert result is fake_harness
    mock_create.assert_called_once_with(
        "Research",
        llm_override="gpt_41mini@openai",
        force_memory_checkpointer=True,
        extra_tools=extra_tools,
    )


async def test_build_harness_without_profile_builds_adhoc_react_harness() -> None:
    from genai_tk.mcp.agent_tool import _build_harness

    cfg = MCPAgentConfig(name="run_adhoc", profile=None, llm="parrot_local@fake")
    extra_tools = [_fake_tool]

    harness = await _build_harness(cfg, extra_tools)

    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    assert isinstance(harness, LangChainHarness)
    assert harness._profile.type == "react"
    assert harness._extra_tools == extra_tools
