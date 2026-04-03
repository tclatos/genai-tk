"""Unit tests for the embedded DeerFlow client event translation."""

from __future__ import annotations

import inspect
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from genai_tk.agents.deer_flow.embedded_client import (
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    _mode_flags,
    _translate_event,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Reusable fake StreamEvent class — avoids importing the real src.client
_FakeStreamEvent = type("StreamEvent", (), {})


@pytest.fixture(autouse=True)
def _fake_src_client_module():
    """Ensure a fake ``src.client`` module with ``StreamEvent`` is always available.

    This prevents tests from accidentally importing the real DeerFlow module
    (which would require DEER_FLOW_PATH on sys.path) and keeps test isolation.
    """
    fake_src = sys.modules.get("src") or ModuleType("src")
    fake_src_client = ModuleType("src.client")
    fake_src_client.StreamEvent = _FakeStreamEvent  # type: ignore[attr-defined]
    prev_src = sys.modules.get("src")
    prev_client = sys.modules.get("src.client")
    sys.modules["src"] = fake_src
    sys.modules["src.client"] = fake_src_client

    # Reset the cached class so each test starts clean
    import genai_tk.agents.deer_flow.embedded_client as _ec

    old_cached = _ec._DFStreamEvent
    _ec._DFStreamEvent = None

    yield _FakeStreamEvent

    # Restore
    _ec._DFStreamEvent = old_cached
    if prev_client is not None:
        sys.modules["src.client"] = prev_client
    else:
        sys.modules.pop("src.client", None)
    if prev_src is not None:
        sys.modules["src"] = prev_src
    else:
        sys.modules.pop("src", None)


def _make_event(type_: str, data: dict):
    """Create a fake DeerFlow StreamEvent with given type and data."""
    ev = _FakeStreamEvent()
    ev.type = type_  # type: ignore[attr-defined]
    ev.data = data  # type: ignore[attr-defined]
    return ev


@pytest.fixture()
def fake_deer_flow_env(monkeypatch, tmp_path):
    """Set up DEER_FLOW_PATH, backend dir, fake src.client with a mock DeerFlowClient."""
    monkeypatch.setenv("DEER_FLOW_PATH", str(tmp_path))
    (tmp_path / "backend").mkdir(exist_ok=True)

    fake_config = tmp_path / "config.yaml"
    fake_config.write_text("models: {}")

    mock_df_client_cls = MagicMock()
    fake_src_client = sys.modules["src.client"]
    fake_src_client.DeerFlowClient = mock_df_client_cls  # type: ignore[attr-defined]

    # Also inject a fake deerflow.client so the modern-layout import path
    # resolves to the same mock (prevents loading the real deer-flow module).
    fake_df = sys.modules.get("deerflow") or ModuleType("deerflow")
    fake_df_client = ModuleType("deerflow.client")
    fake_df_client.DeerFlowClient = mock_df_client_cls  # type: ignore[attr-defined]
    fake_df_client.StreamEvent = _FakeStreamEvent  # type: ignore[attr-defined]
    prev_df = sys.modules.get("deerflow")
    prev_df_client = sys.modules.get("deerflow.client")
    sys.modules["deerflow"] = fake_df
    sys.modules["deerflow.client"] = fake_df_client

    # Reset the compat-check flag so clean-room detection runs per test
    import genai_tk.agents.deer_flow.embedded_client as _ec

    old_compat = _ec._compat_checked
    _ec._compat_checked = True  # skip compat check in unit tests

    yield {"config_path": fake_config, "mock_cls": mock_df_client_cls}

    _ec._compat_checked = old_compat
    if prev_df_client is not None:
        sys.modules["deerflow.client"] = prev_df_client
    else:
        sys.modules.pop("deerflow.client", None)
    if prev_df is not None:
        sys.modules["deerflow"] = prev_df
    else:
        sys.modules.pop("deerflow", None)


# ---------------------------------------------------------------------------
# _translate_event
# ---------------------------------------------------------------------------


def test_translate_ai_text_yields_token_event() -> None:
    """AI message with text content translates to TokenEvent."""
    ev = _make_event("messages-tuple", {"type": "ai", "content": "Hello!", "tool_calls": []})
    result = _translate_event(ev)
    assert len(result) == 1
    assert isinstance(result[0], TokenEvent)
    assert result[0].data == "Hello!"


def test_translate_ai_tool_call_yields_tool_call_event() -> None:
    """AI message with tool_calls translates to ToolCallEvent(s)."""
    ev = _make_event(
        "messages-tuple",
        {
            "type": "ai",
            "content": "",
            "tool_calls": [{"name": "web_search", "args": {"query": "AI"}, "id": "tc1"}],
        },
    )
    result = _translate_event(ev)
    tool_calls = [e for e in result if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "web_search"
    assert tool_calls[0].args == {"query": "AI"}
    assert tool_calls[0].call_id == "tc1"


def test_translate_tool_result_event() -> None:
    """Tool message translates to ToolResultEvent."""
    ev = _make_event(
        "messages-tuple",
        {"type": "tool", "name": "web_search", "content": "Results here", "tool_call_id": "tc1"},
    )
    result = _translate_event(ev)
    assert len(result) == 1
    assert isinstance(result[0], ToolResultEvent)
    assert result[0].tool_name == "web_search"
    assert result[0].content == "Results here"
    assert result[0].call_id == "tc1"


def test_translate_values_event_returns_empty() -> None:
    """'values' events are silently consumed (return empty list)."""
    ev = _make_event("values", {"messages": [], "title": None, "artifacts": []})
    assert _translate_event(ev) == []


def test_translate_end_event_returns_empty() -> None:
    """'end' events are silently consumed."""
    ev = _make_event("end", {})
    assert _translate_event(ev) == []


# ---------------------------------------------------------------------------
# _mode_flags
# ---------------------------------------------------------------------------


def test_mode_flags_flash() -> None:
    flags = _mode_flags("flash")
    assert flags["thinking_enabled"] is False
    assert flags["is_plan_mode"] is False
    assert flags["subagent_enabled"] is False


def test_mode_flags_thinking() -> None:
    flags = _mode_flags("thinking")
    assert flags["thinking_enabled"] is True
    assert flags["is_plan_mode"] is False


def test_mode_flags_pro() -> None:
    flags = _mode_flags("pro")
    assert flags["thinking_enabled"] is True
    assert flags["is_plan_mode"] is True
    assert flags["subagent_enabled"] is False


def test_mode_flags_ultra() -> None:
    flags = _mode_flags("ultra")
    assert flags["thinking_enabled"] is True
    assert flags["is_plan_mode"] is True
    assert flags["subagent_enabled"] is True


def test_mode_flags_unknown_defaults_to_flash() -> None:
    flags = _mode_flags("unknown_mode")
    assert flags == _mode_flags("flash")


# ---------------------------------------------------------------------------
# Event dataclass defaults
# ---------------------------------------------------------------------------


def test_token_event_defaults() -> None:
    ev = TokenEvent()
    assert ev.kind == "token"
    assert ev.data == ""


def test_node_event_state_default() -> None:
    ev = NodeEvent()
    assert ev.state == {}


def test_tool_call_event_args_default() -> None:
    ev = ToolCallEvent()
    assert ev.args == {}


def test_error_event() -> None:
    ev = ErrorEvent(message="boom")
    assert ev.kind == "error"
    assert ev.message == "boom"


# ---------------------------------------------------------------------------
# Signature regression tests — guard against re-introducing removed params
# ---------------------------------------------------------------------------


def test_run_single_shot_no_stream_enabled() -> None:
    """_run_single_shot must not accept stream_enabled (removed in embedded refactor)."""
    from genai_tk.agents.deer_flow.cli_commands import _run_single_shot

    params = inspect.signature(_run_single_shot).parameters
    assert "stream_enabled" not in params, (
        "_run_single_shot still has 'stream_enabled' — it was removed when switching to embedded mode"
    )


def test_run_chat_mode_no_stream_enabled() -> None:
    """_run_chat_mode must not accept stream_enabled (removed in embedded refactor)."""
    from genai_tk.agents.deer_flow.cli_commands import _run_chat_mode

    params = inspect.signature(_run_chat_mode).parameters
    assert "stream_enabled" not in params, (
        "_run_chat_mode still has 'stream_enabled' — it was removed when switching to embedded mode"
    )


def test_run_single_shot_has_expected_params() -> None:
    """_run_single_shot has all expected parameters."""
    from genai_tk.agents.deer_flow.cli_commands import _run_single_shot

    params = inspect.signature(_run_single_shot).parameters
    for expected in (
        "profile_name",
        "user_input",
        "llm_override",
        "extra_mcp",
        "mode_override",
        "verbose",
        "show_trace",
        "subagent_enabled",
        "plan_mode",
    ):
        assert expected in params, f"_run_single_shot is missing expected param '{expected}'"


def test_run_chat_mode_has_expected_params() -> None:
    """_run_chat_mode has all expected parameters."""
    from genai_tk.agents.deer_flow.cli_commands import _run_chat_mode

    params = inspect.signature(_run_chat_mode).parameters
    for expected in (
        "profile_name",
        "llm_override",
        "extra_mcp",
        "mode_override",
        "verbose",
        "show_trace",
        "initial_input",
        "subagent_enabled",
        "plan_mode",
    ):
        assert expected in params, f"_run_chat_mode is missing expected param '{expected}'"


def test_commands_agents_deerflow_no_stream_kwarg() -> None:
    """commands_agents deerflow callback must not pass stream_enabled to cli functions."""
    import ast
    from pathlib import Path

    src = (Path(__file__).parents[4] / "genai_tk" / "agents" / "commands_agents.py").read_text()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert kw.arg != "stream_enabled", (
                    f"commands_agents.py line {node.lineno}: call passes 'stream_enabled' — "
                    "this kwarg was removed in the embedded client refactor"
                )


def test_stream_message_signature() -> None:
    """EmbeddedDeerFlowClient.stream_message has expected keyword parameters."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    params = inspect.signature(EmbeddedDeerFlowClient.stream_message).parameters
    for expected in ("thread_id", "user_input", "model_name", "mode", "subagent_enabled", "plan_mode"):
        assert expected in params, f"stream_message is missing expected param '{expected}'"


# ---------------------------------------------------------------------------
# Checkpointer availability
# ---------------------------------------------------------------------------


def test_sqlite_checkpointer_importable() -> None:
    """langgraph-checkpoint-sqlite package must be installed (added to pyproject.toml)."""
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401


def test_memory_saver_importable() -> None:
    """MemorySaver fallback must always be importable from core langgraph."""
    from langgraph.checkpoint.memory import MemorySaver  # noqa: F401


def test_embedded_client_falls_back_to_memory_saver(fake_deer_flow_env) -> None:
    """EmbeddedDeerFlowClient uses MemorySaver when SqliteSaver is unavailable."""
    from unittest.mock import patch

    from langgraph.checkpoint.memory import MemorySaver

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    with patch.dict(sys.modules, {"langgraph.checkpoint.sqlite": None}):
        client = EmbeddedDeerFlowClient(config_path=fake_deer_flow_env["config_path"])
        assert isinstance(client._checkpointer, MemorySaver)


def test_embedded_client_init_signature() -> None:
    """EmbeddedDeerFlowClient.__init__ accepts config_path and model_name."""
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    params = inspect.signature(EmbeddedDeerFlowClient.__init__).parameters
    assert "config_path" in params
    assert "model_name" in params


# ---------------------------------------------------------------------------
# _resolve_model_name — delegates to resolve_llm_identifier_safe
# ---------------------------------------------------------------------------


def test_resolve_model_name_delegates_to_resolve_llm_identifier() -> None:
    """_resolve_model_name delegates to LlmFactory.resolve_llm_identifier_safe."""
    from unittest.mock import patch

    from genai_tk.agents.deer_flow.cli_commands import _resolve_model_name

    with patch(
        "genai_tk.core.llm_factory.LlmFactory.resolve_llm_identifier_safe",
        return_value=("openai/gpt-oss-120b@openrouter", None),
    ):
        assert _resolve_model_name("gpt_oss120@openrouter") == "openai/gpt-oss-120b@openrouter"


def test_resolve_model_name_error_raises_exit() -> None:
    """Exits with an error when resolution fails."""
    from unittest.mock import patch

    import typer

    from genai_tk.agents.deer_flow.cli_commands import _resolve_model_name

    with patch(
        "genai_tk.core.llm_factory.LlmFactory.resolve_llm_identifier_safe",
        return_value=(None, "Unknown LLM: 'ghost@nowhere'"),
    ):
        with pytest.raises(typer.Exit):
            _resolve_model_name("ghost@nowhere")


# ---------------------------------------------------------------------------
# config_bridge — generate_deer_flow_models + _build_dynamic_model_entry
# ---------------------------------------------------------------------------


def test_build_dynamic_model_entry_openrouter() -> None:
    """_build_dynamic_model_entry builds a valid entry for a gateway model."""
    from unittest.mock import patch

    from genai_tk.agents.deer_flow.config_bridge import _build_dynamic_model_entry

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        entry = _build_dynamic_model_entry("openai/gpt-oss-120b@openrouter")

    assert entry is not None
    assert entry["name"] == "openai/gpt-oss-120b@openrouter"
    assert entry["model"] == "openai/gpt-oss-120b"
    assert "openai_api_base" in entry
    assert entry["api_key"] == "$OPENROUTER_API_KEY"


def test_build_dynamic_model_entry_unknown_provider() -> None:
    """Returns None for an unknown provider."""
    from genai_tk.agents.deer_flow.config_bridge import _build_dynamic_model_entry

    assert _build_dynamic_model_entry("some-model@unknown_provider") is None


def test_build_dynamic_model_entry_missing_api_key() -> None:
    """Returns None when the required API key is not set."""
    from unittest.mock import patch

    from genai_tk.agents.deer_flow.config_bridge import _build_dynamic_model_entry

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        entry = _build_dynamic_model_entry("openai/gpt-oss-120b@openrouter")

    assert entry is None


def test_generate_deer_flow_models_uses_known_items_dict() -> None:
    """generate_deer_flow_models uses known_items_dict, not just known_list."""
    from unittest.mock import MagicMock, patch

    from genai_tk.agents.deer_flow.config_bridge import generate_deer_flow_models

    mock_info = MagicMock()
    mock_info.id = "gpt-4.1-mini@openai"
    mock_info.provider = "openai"
    mock_info.model = "gpt-4.1-mini"
    mock_info.max_tokens = 16384
    mock_info.supports_vision = True
    mock_info.supports_thinking = False
    prov = MagicMock()
    prov.api_key_env_var = "OPENAI_API_KEY"
    prov.api_base = None
    prov.get_use_string.return_value = "langchain_openai.ChatOpenAI"
    mock_info.get_provider_info.return_value = prov

    with (
        patch(
            "genai_tk.agents.deer_flow.config_bridge.LlmFactory.known_items_dict",
            return_value={"gpt-4.1-mini@openai": mock_info},
        ),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test"}),
    ):
        models = generate_deer_flow_models(selected_llm_id="gpt-4.1-mini@openai")

    assert len(models) == 1
    assert models[0]["name"] == "gpt-4.1-mini@openai"


# ---------------------------------------------------------------------------
# End-to-end integration-style tests
# ---------------------------------------------------------------------------


def test_readabilipy_importable() -> None:
    """readabilipy must be installed — required by DeerFlow's jina_ai web_fetch tool."""
    from readabilipy import simple_json_from_html_string  # noqa: F401


def test_sqlite_checkpointer_is_base_saver() -> None:
    """SqliteSaver constructed with a direct connection is a valid BaseCheckpointSaver."""
    import sqlite3
    import tempfile

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = os.path.join(tempfile.mkdtemp(), "test_e2e.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    assert isinstance(saver, BaseCheckpointSaver), f"Expected BaseCheckpointSaver, got {type(saver).__name__}"
    conn.close()


def test_embedded_client_creates_valid_checkpointer(fake_deer_flow_env) -> None:
    """EmbeddedDeerFlowClient.__init__ creates a checkpointer that passes DeerFlow validation."""
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    client = EmbeddedDeerFlowClient(config_path=fake_deer_flow_env["config_path"])
    assert isinstance(client._checkpointer, BaseCheckpointSaver), (
        f"Checkpointer must be BaseCheckpointSaver, got {type(client._checkpointer).__name__}"
    )


def test_sqlite_connection_survives_after_init(fake_deer_flow_env) -> None:
    """Regression: sqlite connection must remain open after __init__ returns.

    The old ``from_conn_string().__enter__()`` approach caused the database
    to close when the generator context manager was garbage-collected.
    """
    import gc

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    client = EmbeddedDeerFlowClient(config_path=fake_deer_flow_env["config_path"])
    gc.collect()  # Force GC to trigger any generator finalization
    client._checkpointer.conn.execute("SELECT 1")  # Connection must still be usable


def test_config_bridge_to_embedded_client_name_consistency() -> None:
    """The model name from _resolve_model_name matches config_bridge output names."""
    from unittest.mock import patch

    from genai_tk.agents.deer_flow.cli_commands import _resolve_model_name
    from genai_tk.agents.deer_flow.config_bridge import generate_deer_flow_models

    # Simulate a model resolved via models.dev (gateway model not in llm.yaml)
    with patch(
        "genai_tk.core.llm_factory.LlmFactory.resolve_llm_identifier_safe",
        return_value=("openai/gpt-oss-120b@openrouter", None),
    ):
        resolved = _resolve_model_name("gpt_oss120@openrouter")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
        models = generate_deer_flow_models(selected_llm_id=resolved)

    assert len(models) == 1
    assert models[0]["name"] == resolved, f"config_bridge name '{models[0]['name']}' must match resolved '{resolved}'"


def test_full_init_chain_no_deer_flow_path(monkeypatch) -> None:
    """EmbeddedDeerFlowClient raises RuntimeError when DEER_FLOW_PATH is unset."""
    monkeypatch.delenv("DEER_FLOW_PATH", raising=False)

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    with pytest.raises(RuntimeError, match="DEER_FLOW_PATH"):
        EmbeddedDeerFlowClient(config_path="/nonexistent/config.yaml")
