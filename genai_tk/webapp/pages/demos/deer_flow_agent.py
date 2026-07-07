"""Streamlit DeerFlow demo — DeerFlow-inspired 2-panel UI.

Layout mirrors DeerFlow's native workspace:
  - Left panel  : Execution trace — structured phase cards with tool rows.
                  Click 📄 on any tool to load its output in the Artifact tab.
  - Right panel : Two tabs — "💬 Chat" (conversation) and "📄 Artifact" (code /
                  files / search results / Mermaid diagrams).

Runs through :class:`~genai_tk.agents.harness.deerflow_harness.DeerFlowHarness`
and the shared :mod:`~genai_tk.webapp.ui_components.harness_workbench` — the
same rendering model used by the ReAct (LangChain) demo page.
"""

import traceback
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from streamlit import session_state as sss

from genai_tk.agents.deer_flow import DeerFlowProfile
from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness
from genai_tk.agents.harness.profiles import load_deerflow_profiles
from genai_tk.webapp.ui_components.agent_layout import render_agent_sidebar, render_sidebar_demo_section
from genai_tk.webapp.ui_components.harness_workbench import (
    render_artifact,
    render_artifact_gallery,
    render_trace_panel,
    stream_harness_turn,
)
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_FILE = "config/agents/deerflow.yaml"
PAGE_TITLE = "🦌 DeerFlow Agent"
CHAT_HEIGHT = 600
TRACE_HEIGHT = 680

MODE_LABELS: dict[str, str] = {
    "flash": "⚡ Flash",
    "thinking": "💡 Thinking",
    "pro": "🎓 Pro",
    "ultra": "🚀 Ultra",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "df_messages": [],
        "df_thread_id": None,
        "df_harness": None,
        "df_profile_name": None,
        "df_active_profile": None,
        "df_mode": "pro",
        "df_model_name": None,
        "df_runtime_signature": None,
        "df_server_ready": False,
        "df_trace_steps": [],  # list[TraceStep]
        "df_all_artifacts": [],  # list[Artifact] from completed run
        "df_selected_artifact": None,  # Artifact | None
        "df_trace_verbose": True,
        "df_error": None,
    }
    for key, val in defaults.items():
        if key not in sss:
            sss[key] = val


def _clear_chat() -> None:
    sss.df_messages = []
    sss.df_thread_id = None
    sss.df_trace_steps = []
    sss.df_all_artifacts = []
    sss.df_selected_artifact = None


def _clear_runtime() -> None:
    sss.df_server_ready = False
    sss.df_harness = None
    sss.df_active_profile = None
    sss.df_model_name = None
    sss.df_runtime_signature = None


def _clear_all() -> None:
    _clear_chat()
    _clear_runtime()
    sss.df_error = None


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_profiles() -> list[DeerFlowProfile]:
    try:
        return load_deerflow_profiles()
    except Exception as exc:
        logger.error("Failed to load DeerFlow profiles: {}", exc)
        return []


def _profile_by_name(profiles: list[DeerFlowProfile], name: str) -> DeerFlowProfile | None:
    return next((p for p in profiles if p.name == name), None)


# ---------------------------------------------------------------------------
# Runtime lifecycle
# ---------------------------------------------------------------------------


def _selected_llm_override() -> str | None:
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().get_str("llm.models.default") or None
    except Exception:
        return None


def _runtime_signature(profile_name: str, llm: str | None) -> str:
    return f"{profile_name}|{llm or ''}"


def _ensure_runtime(profile_name: str) -> tuple[DeerFlowHarness, DeerFlowProfile, str | None]:
    """Prepare the DeerFlowHarness (once per profile+LLM combination) and cache in session state."""
    llm = _selected_llm_override()
    sig = _runtime_signature(profile_name, llm)

    if sss.df_server_ready and sss.df_harness and sss.df_runtime_signature == sig and sss.df_active_profile:
        return sss.df_harness, sss.df_active_profile, sss.df_model_name

    harness = DeerFlowHarness(profile_name, llm_override=llm)
    import asyncio

    asyncio.run(harness.ensure_ready())

    sss.df_server_ready = True
    sss.df_harness = harness
    sss.df_active_profile = harness.profile
    sss.df_model_name = harness.model_name
    sss.df_runtime_signature = sig
    return harness, harness.profile, harness.model_name


# ---------------------------------------------------------------------------
# UI — trace + artifact panels (shared with the ReAct demo page)
# ---------------------------------------------------------------------------
#
# Rendering (render_trace_panel / render_artifact / render_artifact_gallery)
# lives in genai_tk.webapp.ui_components.harness_workbench so both demo pages
# share one visual model.


def _render_sidebar(profiles: list[DeerFlowProfile]) -> tuple[str | None, str]:
    """Render the full DeerFlow sidebar.

    Contains: profile selector + metadata + examples list, mode selectbox,
    verbose trace toggle, and clear buttons.

    Returns:
        ``(selected_profile_name_or_None, selected_mode_key)``
    """
    with st.sidebar:
        st.divider()

        def _profile_info(p: DeerFlowProfile) -> None:
            if p.description:
                st.caption(p.description)
            if p.tool_groups:
                st.markdown("**Tools:** " + ", ".join(f"`{g}`" for g in p.tool_groups))
            if p.mcp_servers:
                st.markdown("**MCP:** " + ", ".join(f"`{m}`" for m in p.mcp_servers))

        selected_profile = render_sidebar_demo_section(
            profiles,
            current_name=sss.df_profile_name,
            info_fn=_profile_info,
        )

        st.divider()

        mode_keys = list(MODE_LABELS.keys())
        mode_idx = mode_keys.index(sss.df_mode) if sss.df_mode in mode_keys else 0
        selected_mode = st.selectbox(
            "⚙️ Mode",
            mode_keys,
            format_func=lambda x: MODE_LABELS[x],
            index=mode_idx,
            key="df_mode_sel",
        )

        sss.df_trace_verbose = st.toggle(
            "Verbose trace",
            value=sss.df_trace_verbose,
            help="Show unlabelled graph nodes.",
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Chat", help="Clear conversation"):
                _clear_chat()
                st.rerun()
        with col2:
            if st.button("🗑️ Full", help="Clear runtime + chat"):
                _clear_all()
                st.rerun()

        st.divider()
        st.caption("Full UI: `cli deerflow --web`")

    selected_name = selected_profile.name if selected_profile else None
    return selected_name, selected_mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_session()
    profiles = _load_profiles()

    # Sidebar: LLM selector + Edit Config (shared) + profile/mode/clear (DeerFlow-specific)
    render_agent_sidebar(CONFIG_FILE)
    selected_name, selected_mode = _render_sidebar(profiles)

    if selected_name is None:
        st.stop()

    # ── Title ─────────────────────────────────────────────────────────────
    st.title(PAGE_TITLE)

    # Detect profile change → clear state
    if sss.df_profile_name and sss.df_profile_name != selected_name:
        _clear_all()

    sss.df_profile_name = selected_name
    sss.df_mode = selected_mode

    profile = _profile_by_name(profiles, selected_name)
    if profile is None:
        st.error(f"Profile '{selected_name}' not found.")
        st.stop()

    if sss.df_error:
        st.error(sss.df_error)
        if st.button("Dismiss"):
            sss.df_error = None
            st.rerun()

    # Two-column layout: [2 trace] | [3 chat + artifact]
    col_trace, col_main = st.columns([2, 3], gap="medium")

    with col_trace:
        st.subheader("🔍 Execution Trace")
        with st.container(height=TRACE_HEIGHT, border=True):
            render_trace_panel(sss.df_trace_steps, key_prefix="df")

    with col_main:
        tab_chat, tab_artifact = st.tabs(["\U0001f4ac Chat", "\U0001f4c4 Artifact"])

        with tab_chat:
            with st.container(height=CHAT_HEIGHT, border=True):
                if not sss.df_messages:
                    st.info("Hello! I'm DeerFlow. Ask anything or pick an example from the sidebar.")
                for msg in sss.df_messages:
                    if msg["role"] == "user":
                        st.chat_message("human").write(msg["content"])
                    else:
                        with st.chat_message("ai"):
                            render_message_with_mermaid(msg["content"], st)

        with tab_artifact:
            with st.container(height=CHAT_HEIGHT, border=True):
                if sss.df_selected_artifact:
                    if st.button("\u2190 All outputs"):
                        sss.df_selected_artifact = None
                        st.rerun()
                    render_artifact(sss.df_selected_artifact)
                else:
                    render_artifact_gallery(sss.df_all_artifacts, key_prefix="df")

    if sss.get("df_show_info"):
        sss.df_show_info = False
        active = sss.df_active_profile or profile
        with st.container(border=True):
            st.markdown(f"**Profile:** `{selected_name}` \u00b7 **Mode:** `{selected_mode}`")
            st.markdown(f"**Model:** `{sss.df_model_name or '(profile default)'}`")
            if active.mcp_servers:
                st.markdown("**MCP:** " + ", ".join(active.mcp_servers))
            if sss.df_thread_id:
                st.markdown(f"**Thread:** `{sss.df_thread_id}`")

    user_input = st.chat_input("Ask DeerFlow… or /help", key="df_input")
    if not user_input or not user_input.strip():
        return

    user_input = user_input.strip()

    # Slash commands
    if user_input.startswith("/"):
        cmd = user_input.lower().strip()
        if cmd in ("/clear", "/reset", "/quit", "/exit", "/q"):
            _clear_chat()
        elif cmd == "/help":
            st.info(
                "**Commands:**\n"
                "- `/mode flash|thinking|pro|ultra` \u2014 switch mode\n"
                "- `/trace` \u2014 toggle verbose trace\n"
                "- `/clear` \u2014 new conversation\n"
                "- `/info` \u2014 show runtime details\n"
                "- `/help` \u2014 this message",
                icon="\U0001f4d6",
            )
        elif cmd == "/info":
            sss.df_show_info = True
        elif cmd == "/trace":
            sss.df_trace_verbose = not sss.df_trace_verbose
            st.info(f"Verbose trace: {'**ON**' if sss.df_trace_verbose else '**OFF**'}")
        elif cmd.startswith("/mode"):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                st.info(f"Current mode: `{sss.df_mode}`")
            else:
                new_mode = parts[1].strip()
                if new_mode in MODE_LABELS:
                    sss.df_mode = new_mode
                    st.success(f"Mode \u2192 `{new_mode}` ({MODE_LABELS[new_mode]})")
                else:
                    st.warning("Unknown mode. Choose: `flash` | `thinking` | `pro` | `ultra`")
        else:
            st.warning(f"Unknown command `{user_input}`. Try `/help`.")
        st.rerun()
        return

    # Agent call
    sss.df_messages.append({"role": "user", "content": user_input})
    sss.df_selected_artifact = None
    sss.df_all_artifacts = []
    sss.df_trace_steps = []

    with st.spinner("\U0001f98c Preparing DeerFlow runtime\u2026"):
        try:
            harness, prepared_profile, model_name = _ensure_runtime(selected_name)
            # The mode selector overrides the profile's configured default mode.
            if prepared_profile is not None:
                prepared_profile.mode = sss.df_mode
        except Exception as exc:
            sss.df_error = f"Failed to prepare DeerFlow runtime: {exc}"
            logger.error("{}\n{}", sss.df_error, traceback.format_exc())
            st.rerun()
            return

    if not sss.df_thread_id:
        sss.df_thread_id = uuid.uuid4().hex

    # Status widget + execution live in the left trace panel.
    # response_placeholder is created inside the status so streaming tokens
    # appear there during the run; after st.rerun() the final answer renders
    # properly in the chat history.
    with col_trace:
        with st.status("🦌 Running…", expanded=True) as status_widget:
            response_placeholder = st.empty()
            try:
                turn = stream_harness_turn(
                    harness,
                    user_input,
                    thread_id=sss.df_thread_id,
                    response_placeholder=response_placeholder,
                )
                if turn.is_clarification:
                    status_widget.update(label="❓ Clarification needed", state="complete", expanded=False)
                else:
                    status_widget.update(label="✅ Done", state="complete", expanded=False)
            except Exception as exc:
                sss.df_error = f"Agent error: {exc}"
                logger.error("{}\n{}", sss.df_error, traceback.format_exc())
                status_widget.update(label="❌ Error", state="error")
                st.rerun()
                return

    response_placeholder.empty()

    sss.df_active_profile = prepared_profile
    sss.df_model_name = model_name

    if not turn.text:
        st.rerun()
        return

    sss.df_trace_steps = turn.steps
    sss.df_messages.append({"role": "assistant", "content": turn.text})

    if not turn.is_clarification:
        tool_artifacts = [tool.artifact for step in turn.steps for tool in step.tools if tool.artifact is not None]
        sss.df_all_artifacts = tool_artifacts + turn.artifacts

    st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

try:
    _ = st.session_state
    main()
except (AttributeError, RuntimeError):
    pass
