"""Shared Streamlit workbench for any :class:`~genai_tk.agents.harness.base.BaseHarness`.

Provides one reusable rendering model — trace phase cards, chat transcript,
and an artifact gallery/viewer — driven entirely by the canonical
:mod:`genai_tk.agents.harness.events` event stream. Both the LangChain and
DeerFlow demo pages consume this module so they share one visual and
behavioural model instead of maintaining two parallel implementations.
"""

from __future__ import annotations

import asyncio
import re
from time import monotonic
from typing import Any

import streamlit as st
from pydantic import BaseModel, Field
from streamlit import session_state as sss

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.events import (
    ClarificationEvent,
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid

# ---------------------------------------------------------------------------
# Node labels / icons (used by DeerFlow, which emits NodeEvent) and a
# tool-name → phase mapping (used by both harnesses to group tool calls into
# a phase when no explicit NodeEvent is emitted, e.g. flat ReAct agents).
# ---------------------------------------------------------------------------

NODE_LABELS: dict[str, str] = {
    # Lead-agent sub-graph nodes
    "planner": "Planning",
    "reporter": "Writing report",
    "researcher": "Researching",
    "coder": "Writing code",
    "model": "Thinking",
    "agent": "Thinking",
    "tools": "Using tools",
    # Tool nodes
    "search_tool": "Searching",
    "web_search": "Searching the web",
    "tavily_search": "Searching (Tavily)",
    "python_repl": "Running code",
    "bash": "Running shell command",
    "file_read": "Reading file",
    "file_write": "Writing file",
    "browser": "Browsing",
    # Subagent nodes
    "subagent": "Running sub-agent",
    "reflection": "Reflecting",
}

NODE_ICONS: dict[str, str] = {
    "planner": "🗺️",
    "reporter": "✍️",
    "researcher": "🔍",
    "coder": "💻",
    "model": "🤔",
    "agent": "🤔",
    "tools": "🛠️",
    "search_tool": "🌐",
    "web_search": "🌐",
    "tavily_search": "🌐",
    "python_repl": "🐍",
    "bash": "⚡",
    "file_read": "📂",
    "file_write": "📝",
    "browser": "🌍",
    "subagent": "🤖",
    "reflection": "🪞",
}

TOOL_TO_NODE: dict[str, str] = {
    "web_search_tool": "researcher",
    "web_fetch_tool": "researcher",
    "image_search_tool": "researcher",
    "view_image": "researcher",
    "web_search": "researcher",
    "tavily_search": "researcher",
    "search_tool": "researcher",
    "run_python_code": "coder",
    "python_repl": "coder",
    "bash": "coder",
    "execute_code": "coder",
    "file_read": "coder",
    "file_write": "coder",
    "task": "subagent",
    "invoke_acp_agent": "subagent",
    "present_files": "reporter",
    "tool_search": "tools",
}


# ---------------------------------------------------------------------------
# Data models for trace + artifacts
# ---------------------------------------------------------------------------


class Artifact(BaseModel):
    """A single extractable output produced during an agent run."""

    type: str  # "code" | "file" | "search" | "web" | "mermaid" | "text"
    title: str
    content: str
    language: str = ""


class ToolDetail(BaseModel):
    """One tool call + result captured from the event stream."""

    name: str
    args: dict = Field(default_factory=dict)
    result: str = ""
    artifact: Artifact | None = None


class TraceStep(BaseModel):
    """One active phase (Planner, Researcher, Coder, Reporter, Agent …)."""

    node: str
    label: str
    icon: str = "→"
    elapsed: float = 0.0
    tools: list[ToolDetail] = Field(default_factory=list)


class TurnResult(BaseModel):
    """Outcome of one streamed conversation turn."""

    text: str = ""
    steps: list[TraceStep] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    is_clarification: bool = False


# ---------------------------------------------------------------------------
# Artifact extraction helpers (framework-agnostic — based on tool name/content)
# ---------------------------------------------------------------------------

_FENCED_CODE_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
_MERMAID_STARTS = (
    "graph ",
    "flowchart ",
    "sequenceDiagram",
    "classDiagram",
    "erDiagram",
    "gantt",
    "mindmap",
)


def lang_from_path(path: str) -> str:
    """Infer syntax-highlighting language from a file path extension."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python",
        "sh": "bash",
        "bash": "bash",
        "js": "javascript",
        "ts": "typescript",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "md": "markdown",
        "sql": "sql",
        "html": "html",
        "css": "css",
    }.get(ext, "text")


def extract_tool_artifact(name: str, args: dict, result: str) -> Artifact | None:
    """Convert a tool call + result into a displayable Artifact, or None."""
    n = (name or "").lower()

    if n in ("python_repl", "coder", "run_python_code", "execute_code"):
        code = args.get("code") or args.get("script") or ""
        if not code and result:
            code = result
        if code:
            return Artifact(type="code", title=f"🐍 Python — {n}", content=code, language="python")

    if n == "bash":
        cmd = args.get("command") or args.get("cmd") or ""
        output = result or ""
        content = f"# command\n{cmd}\n\n# output\n{output}" if cmd else output
        if content.strip():
            return Artifact(type="code", title="⚡ Shell", content=content, language="bash")

    if n == "file_write":
        path = args.get("path") or args.get("file_path") or "file"
        content = args.get("content") or args.get("text") or result or ""
        if content:
            return Artifact(type="file", title=f"📝 {path}", content=content, language=lang_from_path(path))

    if n in ("web_search", "tavily_search", "search_tool", "web_search_tool"):
        if result:
            return Artifact(type="search", title="🌐 Search results", content=result, language="")

    if n == "browser":
        if result:
            return Artifact(type="web", title="🌍 Browser output", content=result, language="")

    return None


def extract_final_artifacts(text: str) -> list[Artifact]:
    """Extract code blocks and Mermaid diagrams from the final assistant message."""
    artifacts: list[Artifact] = []
    for match in _FENCED_CODE_RE.finditer(text):
        lang, code = match.group(1).strip(), match.group(2).strip()
        if not code:
            continue
        if lang == "mermaid" or any(code.lstrip().startswith(s) for s in _MERMAID_STARTS):
            artifacts.append(Artifact(type="mermaid", title="📊 Diagram", content=code, language="mermaid"))
        else:
            label = lang.capitalize() if lang else "Code"
            artifacts.append(Artifact(type="code", title=f"💻 {label}", content=code, language=lang or "text"))
    return artifacts


# ---------------------------------------------------------------------------
# Streaming — drives the shared workbench from any BaseHarness
# ---------------------------------------------------------------------------


def stream_harness_turn(
    harness: BaseHarness,
    user_input: str,
    *,
    thread_id: str | None = None,
    response_placeholder: Any = None,
) -> TurnResult:
    """Run one turn against *harness* and build a :class:`TurnResult`.

    Works identically for :class:`LangChainHarness` and :class:`DeerFlowHarness`
    since both emit the same canonical
    :mod:`genai_tk.agents.harness.events` types — this is the one shared
    event/rendering model used by both demo pages.

    Args:
        harness: Any ready-to-stream harness session.
        user_input: The user's message.
        thread_id: Conversation thread ID.
        response_placeholder: Optional Streamlit placeholder updated live with
            the partial response text (``st.empty()`` or similar).
    """
    steps: list[TraceStep] = []
    current_step: TraceStep | None = None
    pending_tool: ToolDetail | None = None
    token_parts: list[str] = []
    clarification_text: list[str] = []
    t0 = monotonic()

    def _get_or_create_step(node: str) -> TraceStep:
        nonlocal current_step
        label = NODE_LABELS.get(node, node.replace("_", " ").title())
        icon = NODE_ICONS.get(node, "→")
        step = TraceStep(node=node, label=label, icon=icon, elapsed=round(monotonic() - t0, 1))
        steps.append(step)
        current_step = step
        return step

    def _flush_pending() -> None:
        nonlocal pending_tool
        if pending_tool and current_step:
            current_step.tools.append(pending_tool)
        pending_tool = None

    async def _collect() -> None:
        nonlocal current_step, pending_tool
        async for event in harness.astream(user_input, thread_id=thread_id):
            if isinstance(event, NodeEvent):
                if (not current_step) or (event.node != current_step.node):
                    _flush_pending()
                    _get_or_create_step(event.node)

            elif isinstance(event, ToolCallEvent):
                if event.tool_name:
                    node = TOOL_TO_NODE.get(event.tool_name, "tools")
                    if not current_step or current_step.node != node:
                        _flush_pending()
                        _get_or_create_step(node)
                    else:
                        _flush_pending()
                    pending_tool = ToolDetail(name=event.tool_name, args=event.args or {})

            elif isinstance(event, ClarificationEvent):
                if not current_step:
                    _get_or_create_step("planner")
                clarification_text.append(event.question)
                if response_placeholder is not None:
                    response_placeholder.markdown(event.question + " ▌")

            elif isinstance(event, ToolResultEvent):
                if pending_tool and event.tool_name == pending_tool.name:
                    pending_tool.result = event.content or ""
                    pending_tool.artifact = extract_tool_artifact(
                        pending_tool.name, pending_tool.args, pending_tool.result
                    )
                    if current_step:
                        current_step.tools.append(pending_tool)
                    pending_tool = None
                elif event.tool_name:
                    tool = ToolDetail(name=event.tool_name, result=event.content or "")
                    tool.artifact = extract_tool_artifact(tool.name, {}, tool.result)
                    if current_step:
                        current_step.tools.append(tool)

            elif isinstance(event, TokenEvent):
                if not current_step or current_step.node not in ("reporter", "agent"):
                    _flush_pending()
                    _get_or_create_step("reporter")
                token_parts.append(event.text)
                if response_placeholder is not None:
                    response_placeholder.markdown("".join(token_parts) + " ▌")

            elif isinstance(event, ErrorEvent):
                token_parts.append(f"\n\n⚠️ *{event.message}*")
                if current_step:
                    current_step.tools.append(ToolDetail(name="❌ Error", result=event.message))

        _flush_pending()

    asyncio.run(_collect())

    if clarification_text:
        return TurnResult(text=clarification_text[-1], steps=steps, artifacts=[], is_clarification=True)

    full_text = "".join(token_parts)
    return TurnResult(text=full_text, steps=steps, artifacts=extract_final_artifacts(full_text))


# ---------------------------------------------------------------------------
# UI — trace panel
# ---------------------------------------------------------------------------


def render_trace_panel(steps: list[TraceStep], *, key_prefix: str = "trace") -> None:
    """Render the structured execution timeline. Call inside a scrollable container."""
    if not steps:
        st.markdown(
            "<div style='color:#888;padding:2rem 1rem;text-align:center'>"
            "Send a message to see the agent's activity here.</div>",
            unsafe_allow_html=True,
        )
        return

    for step_idx, step in enumerate(steps):
        dim_time = (
            f"<span style='color:#888;font-size:.8em;float:right'>+{step.elapsed:.0f}s</span>" if step.elapsed else ""
        )
        st.markdown(f"**{step.icon} {step.label}**{dim_time}", unsafe_allow_html=True)

        for tool_idx, tool in enumerate(step.tools):
            args_chip = ""
            if tool.args:
                first_val = str(next(iter(tool.args.values())))[:55].replace("\n", " ")
                args_chip = f"<span style='color:#aaa;font-size:.75em;margin-left:6px'>{first_val}</span>"

            has_artifact = tool.artifact is not None
            col_icon, col_name, col_btn = st.columns([0.08, 0.78, 0.14])
            with col_icon:
                st.markdown("⚙️")
            with col_name:
                st.markdown(f"<code style='font-size:.8em'>{tool.name}</code>{args_chip}", unsafe_allow_html=True)
            with col_btn:
                if has_artifact and st.button("📄", key=f"{key_prefix}_art_{step_idx}_{tool_idx}", help="View output"):
                    sss[f"{key_prefix}_selected_artifact"] = tool.artifact
                    st.rerun()

        if step_idx < len(steps) - 1:
            st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #333'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI — artifact panel
# ---------------------------------------------------------------------------


def render_artifact(artifact: Artifact) -> None:
    """Render one artifact. Call inside a container."""
    st.markdown(f"#### {artifact.title}")
    if artifact.type == "mermaid":
        render_message_with_mermaid(f"```mermaid\n{artifact.content}\n```", st)
    elif artifact.type in ("code", "file"):
        st.code(artifact.content, language=artifact.language or "text")
    elif artifact.type == "search":
        st.markdown(artifact.content[:4000])
    elif artifact.type == "web":
        with st.expander("Full browser output", expanded=True):
            st.text(artifact.content[:4000])
    else:
        st.markdown(artifact.content[:5000])


def render_artifact_gallery(artifacts: list[Artifact], *, key_prefix: str = "trace") -> None:
    """Show artifact buttons. Call inside a container."""
    if not artifacts:
        st.markdown(
            "<div style='color:#888;padding:2rem 1rem;text-align:center'>"
            "Click 📄 on a trace tool to view its output here.</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown("**Outputs from this run** — click to open:")
    for idx, art in enumerate(artifacts):
        if st.button(art.title, key=f"{key_prefix}_gallery_{idx}", use_container_width=True):
            sss[f"{key_prefix}_selected_artifact"] = art
            st.rerun()
