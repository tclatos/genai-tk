"""Phase-0 spike: NeMo Relay ATOF capture with the real default model.

Verifies the residual spike item from the design memo: the NeMo Relay
``LangChainCodec`` round-trips a real model response (model name, token usage,
tool calls) into canonical ATOF ``llm`` end events, end-to-end through the
instrumented ``_create_deep_agent``.

Uses the project's real default model (``llm='default'``) and the ``echo``
tool. Gated behind ``--include-real-models`` because it makes a real API call.

API keys are read from ``~/.env`` (the same file the CLI loads at startup).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.agents.langchain.factory import _create_deep_agent
from genai_tk.core.factories.llm_factory import get_llm
from genai_tk.utils.nemo_relay_setup import (
    flush_nemo_relay_async,
    get_relay_callback_handler,
    reset_nemo_relay,
    setup_nemo_relay,
)
from genai_tk.utils.tracing import reset_monitoring

pytest.importorskip("nemo_relay")

# Real API calls can take a while; allow generous headroom.
_REAL_TIMEOUT = 180


@pytest.fixture(scope="session", autouse=True)
def _load_home_env() -> None:
    """Load ``~/.env`` so provider API keys are available.

    The CLI loads this at startup, but pytest does not go through the CLI, so
    replicate it here for direct ``uv run pytest`` invocation.
    """
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env, override=False)


@tool
def echo(message: str) -> str:
    """Echo the message back unchanged."""
    return f"echo:{message}"


def _llm_end_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ATOF llm scope-end events."""
    return [
        e
        for e in events
        if e.get("kind") == "scope" and e.get("scope_category") == "end" and e.get("category") == "llm"
    ]


def _annotated(e: dict[str, Any]) -> dict[str, Any]:
    """Return the ``annotated_response`` payload of an llm end event."""
    return (e.get("category_profile") or {}).get("annotated_response") or {}


@pytest.mark.real_models
@pytest.mark.asyncio
@pytest.mark.timeout(_REAL_TIMEOUT)
async def test_deep_agent_real_model_atof_codec_roundtrip(tmp_path: Path) -> None:
    """A real-model deep-agent run captures ATOF with model name, usage, tool calls.

    Confirms the ``LangChainCodec`` round-trips the real model response into
    canonical ATOF ``llm`` end events: ``annotated_response.model`` (real model
    name), ``annotated_response.usage`` (token counts), and ``tool_calls`` —
    plus an ``agent`` scope and a ``tool`` scope for ``echo``.
    """
    reset_monitoring()
    reset_nemo_relay()
    try:
        atof_path = tmp_path / "events.jsonl"
        assert setup_nemo_relay(atof_path=atof_path), "nemo_relay subscriber did not activate"

        model = get_llm(llm="default", streaming=False)
        profile = AgentProfileConfig(
            name="relay-real",
            type="deep",
            llm="default",
            tools=[],
            mcp_servers=[],
            skill_directories=[],
            # Disable planning/filesystem to keep the run to model + tool calls
            # only (no backend dependency, fewer LLM calls).
            enable_planning=False,
            enable_file_system=False,
        )
        agent = await _create_deep_agent(model, [echo], MemorySaver(), profile, middlewares=[], backend=None)

        handler = get_relay_callback_handler()
        assert handler is not None, "Relay callback handler unavailable"

        result = await agent.ainvoke(
            {"messages": "Please call the echo tool with the message 'hello', then state the result."},
            config={"configurable": {"thread_id": "1"}, "callbacks": [handler]},
        )
        await flush_nemo_relay_async()

        # The run produced a final assistant message.
        messages = result.get("messages", []) if isinstance(result, dict) else []
        assert messages, "agent run returned no messages"

        lines = [ln for ln in atof_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines, f"no ATOF events written to {atof_path}"
        events = [json.loads(ln) for ln in lines]

        # Scope hierarchy: agent + llm + tool scopes were emitted.
        scope_cats = {
            e.get("category") for e in events if e.get("kind") == "scope" and e.get("scope_category") == "start"
        }
        assert "agent" in scope_cats, f"expected an agent scope, got {scope_cats}"
        assert "llm" in scope_cats, f"expected an llm scope, got {scope_cats}"
        assert "tool" in scope_cats, f"expected a tool scope, got {scope_cats}"

        # Codec round-trip: at least one llm end event carries a real model name
        # and real token usage (the residual spike item from the design memo).
        llm_ends = _llm_end_events(events)
        assert llm_ends, "no llm end events captured"
        with_model = [e for e in llm_ends if _annotated(e).get("model")]
        assert with_model, "no llm end event carried an annotated_response.model"
        with_usage = [
            e
            for e in llm_ends
            if isinstance(_annotated(e).get("usage"), dict)
            and _annotated(e)["usage"].get("prompt_tokens", 0) > 0
            and _annotated(e)["usage"].get("completion_tokens", 0) > 0
        ]
        assert with_usage, "no llm end event carried non-zero token usage"

        # The echo tool call was captured as a tool scope and as an annotated
        # tool_call on an llm end event.
        tool_names = {
            e.get("name")
            for e in events
            if e.get("kind") == "scope" and e.get("scope_category") == "start" and e.get("category") == "tool"
        }
        assert "echo" in tool_names, f"expected an echo tool scope, got {tool_names}"
        echo_calls = [
            tc
            for e in llm_ends
            for tc in (_annotated(e).get("tool_calls") or [])
            if isinstance(tc, dict) and tc.get("name") == "echo"
        ]
        assert echo_calls, "no echo tool_call captured on an llm end event"
    finally:
        reset_nemo_relay()
        reset_monitoring()
