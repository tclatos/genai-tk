"""Streamlit page for the unified Agent demo.

One page for every agent profile — LangChain (react | deep | custom, including
the DeepAgents SDK) and DeerFlow — selected from the unified ``agents:`` config.
A ``st.pills`` filter narrows the profile list by kind (React / DeepAgent /
Custom / DeerFlow). The selected profile runs through the shared harness layer
(:func:`genai_tk.agents.harness.create_harness`) and the shared
:mod:`genai_tk.webapp.ui_components.harness_workbench` two-panel layout
(execution trace + chat/artifact), so there is one visual and event model
instead of one page per runtime.
"""

import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as sss

from genai_tk.agents.harness import create_harness, profile_kind
from genai_tk.agents.harness.base import BaseHarness
from genai_tk.webapp.ui_components.agent_layout import PANEL_HEIGHT, render_agent_sidebar, render_sidebar_monitoring
from genai_tk.webapp.ui_components.harness_workbench import (
    render_artifact,
    render_artifact_gallery,
    render_trace_panel,
    stream_harness_turn,
)
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid

load_dotenv()

CONFIG_FILE = "config/agents.yaml"
CHAT_HEIGHT = PANEL_HEIGHT
TRACE_HEIGHT = PANEL_HEIGHT

# Fixed display order for the kind pills.
_KIND_ORDER = ["React", "DeepAgent", "Custom", "DeerFlow"]
MODE_LABELS: dict[str, str] = {
    "flash": "⚡ Flash",
    "thinking": "💡 Thinking",
    "pro": "🎓 Pro",
    "ultra": "🚀 Ultra",
}


def initialize_session_state() -> None:
    """Initialize session state variables."""
    defaults: dict[str, Any] = {
        "messages": [],
        "selected_key": None,
        "selected_mode": "pro",
        "harness": None,
        "harness_signature": None,
        "thread_id": None,
        "trace_steps": [],
        "all_artifacts": [],
        "selected_artifact": None,
        "error": None,
    }
    for key, val in defaults.items():
        if key not in sss:
            sss[key] = val


def clear_chat_history(keep_traces: bool = False) -> None:
    """Reset the chat history and related state.

    Args:
        keep_traces: If True, preserve execution traces while clearing chat messages.
    """
    sss.messages = []
    sss.thread_id = None
    if not keep_traces:
        sss.trace_steps = []
        sss.all_artifacts = []
        sss.selected_artifact = None


def clear_all_history() -> None:
    """Reset both chat and execution trace history and drop the cached harness."""
    clear_chat_history(keep_traces=False)
    sss.harness = None
    sss.harness_signature = None


def _load_profiles() -> tuple[dict[str, Any], str]:
    """Load every profile keyed by slug, plus the default profile key."""
    from genai_tk.agents.harness.profiles import load_agent_profiles

    profiles, _defaults, default_key = load_agent_profiles()
    return profiles, default_key or ""


def _selected_llm_override() -> str | None:
    """Return the sidebar LLM selector's current default, or None."""
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().get_str("llm.models.default") or None
    except Exception:
        return None


def _profile_info(profile: Any) -> None:
    """Render per-profile metadata (tools/MCP) below the selector."""
    parts: list[str] = []
    if getattr(profile, "harness", None) == "deerflow":
        if profile.tool_groups:
            parts.append("Tools: " + ", ".join(f"`{g}`" for g in profile.tool_groups))
    else:
        if profile.tools:
            tool_names = [
                getattr(t, "tool_class", None) or getattr(t, "function", None) or getattr(t, "factory", "?")
                for t in profile.tools
            ]
            parts.append("Tools: " + ", ".join(f"`{n}`" for n in tool_names))
    if profile.mcp_servers:
        parts.append("MCP: " + ", ".join(f"`{m}`" for m in profile.mcp_servers))
    if parts:
        st.caption("  \n".join(parts))


@st.dialog("Profile Configuration")
def _show_profile_config_dialog(profile: Any) -> None:
    """Modal dialog showing the YAML dump of a profile."""
    import yaml as _yaml

    try:
        raw = profile.model_dump(exclude_none=True, exclude_unset=False)
        st.code(_yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False), language="yaml")
    except Exception as exc:
        st.text(str(exc))


def _init_llm_from_profile(profile: Any) -> None:
    """Set the LLM selector to the model specified in the agent profile."""
    profile_llm: str | None = getattr(profile, "llm", None)
    if not profile_llm:
        return
    from genai_tk.webapp.ui_components.llm_selector import set_active_llm

    set_active_llm(profile_llm)


def _harness_signature(key: str, llm: str | None, mode: str | None) -> str:
    return f"{key}|{llm or ''}|{mode or ''}"


def _ensure_harness(key: str, profile: Any, mode: str | None) -> BaseHarness:
    """Get or create the cached harness for (key, llm, mode)."""
    llm = _selected_llm_override() if profile.harness == "deerflow" else None
    sig = _harness_signature(key, llm, mode)
    if sss.harness is not None and sss.harness_signature == sig:
        return sss.harness

    harness = create_harness(
        key,
        llm_override=llm,
        mode_override=mode if profile.harness == "deerflow" else None,
        force_memory_checkpointer=True,
    )
    # Eagerly prepare adapters that expose ensure_ready() (DeerFlow) so the
    # profile/model are available for display before the first turn.
    ensure_ready = getattr(harness, "ensure_ready", None)
    if callable(ensure_ready):
        import asyncio

        asyncio.run(ensure_ready())
    sss.harness = harness
    sss.harness_signature = sig
    return harness


def handle_command(command: str) -> bool:
    """Handle special commands like /trace, /help, etc.

    Returns:
        True if command was handled, False otherwise.
    """
    command = command.strip().lower()
    if command in ["/quit", "/exit", "/q"]:
        st.info("👋 To quit, simply close this browser tab or navigate away.")
        return True
    elif command == "/help":
        st.info(
            """
            **Available Commands:**
            - `/help` - Show this help message
            - `/trace` - Open last LangSmith trace in browser (if available)
            - `/clear` - Clear chat history
            - `/quit` - Instructions to quit

            **Tips:**
            - Type normally to chat with the agent
            - Use the sidebar to filter by kind and change profile
            - Tool calls appear in the execution trace panel
            """
        )
        return True
    elif command == "/trace":
        st.info("Opening LangSmith traces...")
        st.link_button("🔗 Open Traces", "https://smith.langchain.com/")
        return True
    elif command == "/clear":
        clear_chat_history()
        st.success("Chat history cleared!")
        st.rerun()
        return True
    elif command.startswith("/"):
        st.error(f"Unknown command: {command}. Type `/help` for available commands.")
        return True
    return False


def main() -> None:
    """Main entry point for the unified agent demo page."""
    initialize_session_state()
    profiles, default_key = _load_profiles()
    if not profiles:
        st.error("No agent profiles found.")
        st.stop()

    # (key, profile, kind) triples, preserving config order.
    items = [(k, p, profile_kind(p)) for k, p in profiles.items()]
    kind_of = {k: kind for k, _p, kind in items}
    kinds_present = [k for k in _KIND_ORDER if any(kind == k for _kk, _pp, kind in items)]

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        # ── Profile section ───────────────────────────────────────────────
        st.subheader("Profile")

        selected_kinds = st.pills(
            "Type",
            options=kinds_present,
            default=kinds_present,
            selection_mode="multi",
            key="agent_kind_filter",
        )

        filtered = [(k, p, kind) for k, p, kind in items if kind in selected_kinds] if selected_kinds else []
        if not filtered:
            st.warning("No profiles match the selected filter.")
            st.stop()

        keys = [k for k, _p, _kind in filtered]
        # Keep the previous selection valid; otherwise prefer the default key, else first.
        prev = sss.selected_key if sss.selected_key in keys else (default_key if default_key in keys else keys[0])
        selected_key = st.selectbox(
            "Agent",
            options=keys,
            index=keys.index(prev),
            format_func=lambda k: f"{k} · {kind_of[k]}",
            key="agent_profile_sel",
        )
        profile = profiles[selected_key]

        # Detect profile change (or first load) *before* rendering the LLM
        # selector below, so the Provider/Lab/Model widgets and expander
        # title reflect the profile's LLM immediately. If the profile has no
        # explicit ``llm`` override, the app-level default is left untouched.
        first_load = sss.selected_key is None
        profile_changed = not first_load and sss.selected_key != selected_key
        if profile_changed:
            clear_chat_history(keep_traces=True)
            sss.harness = None
            sss.harness_signature = None
        if first_load or profile_changed:
            _init_llm_from_profile(profile)
        sss.selected_key = selected_key

        _profile_info(profile)

        examples: list[str] = getattr(profile, "examples", None) or []
        if examples:
            with st.expander("💡 Examples", expanded=False):
                for ex in examples:
                    st.code(ex, language="")

        if st.button("⚙️ Profile Config", use_container_width=True):
            _show_profile_config_dialog(profile)

        # DeerFlow mode selector (only for deerflow profiles).
        if profile.harness == "deerflow":
            st.divider()
            mode_keys = list(MODE_LABELS.keys())
            mode_idx = mode_keys.index(sss.selected_mode) if sss.selected_mode in mode_keys else 0
            sss.selected_mode = st.selectbox(
                "⚙️ Mode",
                mode_keys,
                format_func=lambda x: MODE_LABELS[x],
                index=mode_idx,
                key="agent_mode_sel",
            )
        else:
            sss.selected_mode = "pro"  # unused for langchain

        # Detect DeerFlow mode change → reset harness (keep traces).
        if sss.get("last_mode") != sss.selected_mode:
            clear_chat_history(keep_traces=True)
            sss.harness = None
            sss.harness_signature = None
        sss.last_mode = sss.selected_mode

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Chat", help="Clear conversation"):
                clear_chat_history(keep_traces=True)
                st.rerun()
        with c2:
            if st.button("🗑️ All", help="Clear conversation + traces + cached agent"):
                clear_all_history()
                st.rerun()

        # 2. LLM selection (collapsed by default)
        st.divider()
        render_agent_sidebar(CONFIG_FILE)

    # 3. Monitoring (bottom of sidebar)
    render_sidebar_monitoring()

    # ── Title ─────────────────────────────────────────────────────────────
    st.title("🤖 Agent")

    if sss.error:
        st.error(sss.error)
        if st.button("Dismiss"):
            sss.error = None
            st.rerun()

    # ── Two-panel main layout: trace | chat + artifact ───────────────────
    col_trace, col_main = st.columns([2, 3], gap="medium")

    with col_trace:
        st.subheader("🔍 Execution Trace")
        with st.container(height=TRACE_HEIGHT, border=True):
            render_trace_panel(sss.trace_steps, key_prefix="agent")

    with col_main:
        tab_chat, tab_artifact = st.tabs(["💬 Chat", "📄 Artifact"])

        with tab_chat:
            with st.container(height=CHAT_HEIGHT, border=True):
                if not sss.messages:
                    st.info("Hello! I'm your agent. Pick a profile from the sidebar and ask me anything.")
                for msg in sss.messages:
                    if msg["role"] == "user":
                        st.chat_message("human").write(msg["content"])
                    else:
                        with st.chat_message("ai"):
                            render_message_with_mermaid(msg["content"], st)

        with tab_artifact:
            with st.container(height=CHAT_HEIGHT, border=True):
                if sss.selected_artifact:
                    if st.button("← All outputs"):
                        sss.selected_artifact = None
                        st.rerun()
                    render_artifact(sss.selected_artifact)
                else:
                    render_artifact_gallery(sss.all_artifacts, key_prefix="agent")

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Type your message… (or /help)", key="chat_input")
    if not user_input:
        return
    user_input = user_input.strip()

    if handle_command(user_input):
        return

    sss.messages.append({"role": "user", "content": user_input})
    sss.selected_artifact = None

    if not sss.thread_id:
        sss.thread_id = uuid.uuid4().hex

    mode_override = sss.selected_mode if profile.harness == "deerflow" else None
    harness = _ensure_harness(selected_key, profile, mode_override)

    with col_trace:
        with st.status("🤖 Running…", expanded=True) as status_widget:
            response_placeholder = st.empty()
            try:
                turn = stream_harness_turn(
                    harness,
                    user_input,
                    thread_id=sss.thread_id,
                    response_placeholder=response_placeholder,
                )
                status_widget.update(label="✅ Done", state="complete", expanded=False)
            except Exception as exc:
                sss.error = f"Agent error: {exc}"
                status_widget.update(label="❌ Error", state="error")
                st.rerun()
                return

    response_placeholder.empty()

    if not turn.text:
        st.rerun()
        return

    sss.trace_steps = turn.steps
    sss.messages.append({"role": "assistant", "content": turn.text})
    tool_artifacts = [tool.artifact for step in turn.steps for tool in step.tools if tool.artifact is not None]
    sss.all_artifacts = tool_artifacts + turn.artifacts

    st.rerun()


try:
    _ = st.session_state
    main()
except (AttributeError, RuntimeError):
    pass
