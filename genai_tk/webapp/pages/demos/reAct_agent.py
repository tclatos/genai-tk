"""Streamlit page for the ReAct Agent demo.

Provides an interactive chat interface for LangChain agents (react | deep |
custom, including DeepAgents SDK profiles) with a two-panel layout: execution
trace + chat/artifact tabs. Supports demo presets, MCP servers, and slash
commands.

Runs through :class:`~genai_tk.agents.harness.langchain_harness.LangChainHarness`
and the shared :mod:`~genai_tk.webapp.ui_components.harness_workbench` — the
same rendering model used by the DeerFlow demo page, so both pages share one
visual and event model instead of two parallel implementations.
"""

import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as sss

from genai_tk.agents.harness.langchain_harness import LangChainHarness
from genai_tk.agents.harness.profiles import load_langchain_profiles
from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.webapp.ui_components.agent_layout import (
    PANEL_HEIGHT,
    render_agent_sidebar,
    render_sidebar_demo_section,
)
from genai_tk.webapp.ui_components.harness_workbench import (
    render_artifact,
    render_artifact_gallery,
    render_trace_panel,
    stream_harness_turn,
)
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid

load_dotenv()


# Edit-Config target: the project-level unified agents file (best-effort).
CONFIG_FILE = "config/agents.yaml"

CHAT_HEIGHT = PANEL_HEIGHT
TRACE_HEIGHT = PANEL_HEIGHT


def initialize_session_state() -> None:
    """Initialize session state variables."""
    defaults: dict[str, Any] = {
        "messages": [],  # list[dict] with "role" / "content" keys
        "current_demo": None,
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
    """Reset both chat and execution trace history."""
    clear_chat_history(keep_traces=False)


def _react_demo_info(demo: AgentProfileConfig) -> None:
    """Render per-demo metadata (tools, MCP) inside the page header."""
    parts = []
    if demo.tools:
        tool_names = [
            getattr(t, "tool_class", None) or getattr(t, "function", None) or getattr(t, "factory", "?")
            for t in demo.tools
        ]
        parts.append("**Tools:** " + ", ".join(f"`{n}`" for n in tool_names))
    if demo.mcp_servers:
        parts.append("**MCP:** " + ", ".join(f"`{m}`" for m in demo.mcp_servers))
    if parts:
        st.markdown("  \n".join(parts))


def _ensure_harness(demo: AgentProfileConfig) -> LangChainHarness:
    """Get or create the cached harness for the current demo (once per demo)."""
    if sss.harness is not None and sss.harness_signature == demo.name:
        return sss.harness

    harness = LangChainHarness(demo, force_memory_checkpointer=True)
    sss.harness = harness
    sss.harness_signature = demo.name
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
        st.info("""
        **Available Commands:**
        - `/help` - Show this help message
        - `/trace` - Open last LangSmith trace in browser (if available)
        - `/clear` - Clear chat history
        - `/quit` - Instructions to quit

        **Tips:**
        - Type normally to chat with the agent
        - Use the sidebar to change demo configurations
        - Tool calls appear in the execution trace panel
        """)
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
    """Main entry point for the ReAct agent demo page."""
    initialize_session_state()

    sample_demos = list(load_langchain_profiles().values())
    if not sample_demos:
        st.error("No LangChain agent profiles found.")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_agent_sidebar(CONFIG_FILE)
    with st.sidebar:
        st.divider()
        demo = render_sidebar_demo_section(
            sample_demos,
            current_name=sss.current_demo,
            info_fn=_react_demo_info,
        )
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Chat", help="Clear conversation"):
                clear_chat_history(keep_traces=True)
                st.rerun()
        with c2:
            if st.button("🗑️ All", help="Clear conversation + traces"):
                clear_all_history()
                st.rerun()

    if demo is None:
        st.stop()

    # ── Title ─────────────────────────────────────────────────────────────
    st.title("🤖 ReAct Agent")

    # Detect demo change → reset harness (keep traces)
    if sss.current_demo and sss.current_demo != demo.name:
        clear_chat_history(keep_traces=True)
        sss.harness = None
        sss.harness_signature = None
    sss.current_demo = demo.name

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
            render_trace_panel(sss.trace_steps, key_prefix="react")

    with col_main:
        tab_chat, tab_artifact = st.tabs(["💬 Chat", "📄 Artifact"])

        with tab_chat:
            with st.container(height=CHAT_HEIGHT, border=True):
                if not sss.messages:
                    st.info("Hello! I'm your ReAct agent. How can I help you today?")
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
                    render_artifact_gallery(sss.all_artifacts, key_prefix="react")

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

    harness = _ensure_harness(demo)

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
