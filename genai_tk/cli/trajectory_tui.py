"""Interactive Textual TUI for navigating agent execution trajectories.

Provides a rich two-pane explorer:
- Left pane: Hierarchical tree of runs and turns (intertwined LLM calls and tool executions).
- Right pane: Real-time detail inspector with formatted Markdown/code for thoughts,
  tool arguments, logs, and token metrics.
"""

from __future__ import annotations

import json
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.visual import VisualType
from textual.widgets import Footer, Header, Markdown, Select, Static, Tree

from genai_tk.extra.monitoring.trajectory_store import (
    LlmCall,
    SkillLoad,
    ToolCall,
    Trajectory,
    TrajectoryStore,
    TrajectoryTurn,
    short_model_name,
)


def _snippet(text: str | None, limit: int = 60) -> str:
    """Collapse whitespace and truncate a string for tooltip display."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _turn_tooltip(lc: LlmCall) -> str:
    """Tooltip for an LLM turn node."""
    usage = lc.usage or {}
    return "\n".join(
        [
            f"🧠 {lc.model or '?'}",
            f"Tokens: {usage.get('prompt_tokens') or 0} in / {usage.get('completion_tokens') or 0} out",
            f"{lc.started_at} → {lc.ended_at or '?'}",
        ]
    )


def _tool_tooltip(tc: ToolCall) -> str:
    """Tooltip for a tool-execution node."""
    parts = [f"🛠️ {tc.name}"]
    if tc.args:
        parts.append(f"Args: {_snippet(json.dumps(tc.args), 120)}")
    if tc.result:
        parts.append(f"Result: {_snippet(tc.result, 160)}")
    if tc.tool_call_id:
        parts.append(f"Call: {tc.tool_call_id}")
    return "\n".join(parts)


def _skill_tooltip(sl: SkillLoad) -> str:
    """Tooltip for a skill.load node."""
    parts = [f"📚 {sl.skill_name}"]
    if sl.source:
        parts.append(f"Source: {sl.source}")
    if sl.tool_name:
        parts.append(f"Tool: {sl.tool_name}")
    parts.append(f"At: {sl.timestamp}")
    return "\n".join(parts)


class TooltipTree(Tree):
    """Tree that shows a tooltip for the node currently under the mouse.

    Textual re-reads ``widget.tooltip`` on every mouse-move tooltip timer tick,
    so the getter derives the content dynamically from ``hover_line``. Tooltip
    text is stored per node under the ``"tooltip"`` key of the node's data.
    """

    @property
    def tooltip(self) -> VisualType | None:  # type: ignore[override]
        """Tooltip content for the hovered node, or None when not hovering."""
        if self.hover_line < 0:
            return None
        node = self._get_node(self.hover_line)
        if node is None or not isinstance(node.data, dict):
            return None
        tooltip = node.data.get("tooltip")
        return str(tooltip) if tooltip is not None else None

    @tooltip.setter
    def tooltip(self, tooltip: VisualType | None) -> None:
        """Accept (unused) static tooltip assignment for Widget API parity."""
        self._tooltip = tooltip


class TrajectoryTuiApp(App[None]):
    """Textual interactive trajectory navigator."""

    TITLE = "GenAI Toolkit · Trajectory Navigator"
    SUB_TITLE = "ATOF Agent Execution Explorer"

    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }

    #top-bar {
        height: 3;
        background: $panel;
        padding: 0 1;
        align: left middle;
        border-bottom: solid $primary;
    }

    #run-select {
        width: 48;
    }

    #summary-stats {
        padding-left: 2;
        color: $text;
        text-style: bold;
    }

    #main-container {
        height: 1fr;
    }

    #left-panel {
        width: 38%;
        border-right: solid $primary;
        background: $surface;
    }

    #right-panel {
        width: 62%;
        padding: 1 2;
        background: $surface-darken-1;
    }

    Tree {
        background: transparent;
        padding: 1;
    }

    #detail-markdown {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("tab", "toggle_focus", "Switch Pane", show=True),
        Binding("r", "reload_run", "Reload", show=True),
        Binding("d", "toggle_dark", "Dark Mode", show=False),
    ]

    def __init__(
        self,
        initial_run_id: str | None = None,
        store: TrajectoryStore | None = None,
    ) -> None:
        super().__init__()
        self.store = store if store is not None else TrajectoryStore()
        self.initial_run_id = initial_run_id
        self.current_trajectory: Trajectory | None = None
        self.runs = self.store.list_runs()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Build select options from existing runs
        select_options = []
        for r in self.runs:
            short_id = r.run_id[:8] if len(r.run_id) >= 8 else r.run_id
            label = f"{short_id} · {r.profile} ({r.started_at[:16]})"
            select_options.append((label, r.run_id))

        default_val = Select.BLANK
        if self.initial_run_id and any(r.run_id == self.initial_run_id for r in self.runs):
            default_val = self.initial_run_id
        elif self.runs:
            default_val = self.runs[0].run_id

        with Horizontal(id="top-bar"):
            if select_options:
                yield Select(
                    select_options,
                    value=default_val,
                    allow_blank=False,
                    id="run-select",
                    prompt="Select a trajectory run",
                )
            else:
                yield Static("[yellow]No runs found in store[/yellow]", id="run-select")
            yield Static("", id="summary-stats")

        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield TooltipTree("Execution Timeline", id="timeline-tree")
            with VerticalScroll(id="right-panel"):
                yield Markdown("", id="detail-markdown")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the view once widgets are mounted."""
        if self.runs:
            target_id = self.initial_run_id or self.runs[0].run_id
            await self.load_run(target_id)
        else:
            await self.query_one("#detail-markdown", Markdown).update(
                "# No Trajectories Found\n\nNo recorded runs were found in the trajectory store."
            )

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle run dropdown selection changes."""
        if event.value and event.value != Select.BLANK:
            await self.load_run(str(event.value))

    def action_toggle_focus(self) -> None:
        """Toggle focus between tree and detail panels."""
        tree = self.query_one("#timeline-tree", Tree)
        details = self.query_one("#right-panel", VerticalScroll)
        if tree.has_focus:
            details.focus()
        else:
            tree.focus()

    async def action_reload_run(self) -> None:
        """Reload the current run from disk."""
        if self.current_trajectory:
            await self.load_run(self.current_trajectory.run_id)

    async def load_run(self, run_id: str) -> None:
        """Load and populate a trajectory into the UI."""
        traj = self.store.get(run_id)
        if traj is None:
            await self.query_one("#detail-markdown", Markdown).update(
                f"# Run Not Found\n\nRun `{run_id}` could not be loaded from store."
            )
            return

        self.current_trajectory = traj

        # Update summary bar
        status_color = "green" if traj.status == "ok" else "red"
        stats_text = (
            f" [bold {status_color}]● {traj.status.upper()}[/]  "
            f"·  Profile: [cyan]{traj.profile}[/]  "
            f"·  LLMs: [blue]{len(traj.llm_calls)}[/]  "
            f"·  Tools: [magenta]{len(traj.tool_calls)}[/]  "
            f"·  Tokens: [yellow]{traj.total_prompt_tokens:,}[/] in / [yellow]{traj.total_completion_tokens:,}[/] out"
        )
        self.query_one("#summary-stats", Static).update(stats_text)

        # Build timeline tree
        tree = self.query_one("#timeline-tree", Tree)
        tree.clear()
        tree.root.set_label(f"🚀 [bold]{traj.profile}[/] [dim]({traj.run_id[:8]})[/]")
        tree.root.data = {
            "type": "root",
            "traj": traj,
            "tooltip": (
                f"{traj.profile}\nStatus: {traj.status}\nStarted: {traj.started_at}\n"
                f"Tokens: {traj.total_prompt_tokens:,} in / {traj.total_completion_tokens:,} out"
            ),
        }

        # 1. Overview Node
        overview_tooltip = (
            f"{len(traj.llm_calls)} LLM calls · {len(traj.tool_calls)} tool calls · "
            f"{len(traj.skill_loads)} skills loaded\n"
            f"Tokens: {traj.total_prompt_tokens:,} in / {traj.total_completion_tokens:,} out"
        )
        overview_node = tree.root.add(
            "📋 Run Overview", data={"type": "overview", "traj": traj, "tooltip": overview_tooltip}
        )

        # 2. User Prompt Node
        user_msg = self.store._root_user_message(traj)
        if user_msg:
            tree.root.add(
                "👤 User Request", data={"type": "user_msg", "text": user_msg, "tooltip": _snippet(user_msg, 400)}
            )

        # 3. Turns (Intertwined LLMs & Tools)
        turns = traj.turns
        for turn in turns:
            lc = turn.llm_call
            if lc is not None:
                m_short = short_model_name(lc.model)
                usage = lc.usage or {}
                tin = usage.get("prompt_tokens") or 0
                tout = usage.get("completion_tokens") or 0
                tok_str = f" [dim]({tin}/{tout})[/]" if (tin or tout) else ""

                turn_label = f"Turn {turn.index}: [blue]llm[/] [bold]{m_short}[/]{tok_str}"
                turn_node = tree.root.add(
                    turn_label, data={"type": "turn", "turn": turn, "llm": lc, "tooltip": _turn_tooltip(lc)}
                )

                for tc in turn.tool_calls:
                    tool_label = f"🛠️ [magenta]tool[/] [bold]{tc.name}[/]"
                    turn_node.add(tool_label, data={"type": "tool", "tool": tc, "tooltip": _tool_tooltip(tc)})

                for sl in turn.skill_loads:
                    skill_label = f"📚 [yellow]skill.load[/] {sl.skill_name}"
                    turn_node.add(skill_label, data={"type": "skill", "skill": sl, "tooltip": _skill_tooltip(sl)})
            else:
                for tc in turn.tool_calls:
                    tree.root.add(
                        f"🛠️ [magenta]tool[/] [bold]{tc.name}[/]",
                        data={"type": "tool", "tool": tc, "tooltip": _tool_tooltip(tc)},
                    )
                for sl in turn.skill_loads:
                    tree.root.add(
                        f"📚 [yellow]skill.load[/] {sl.skill_name}",
                        data={"type": "skill", "skill": sl, "tooltip": _skill_tooltip(sl)},
                    )

        # 4. Final Response Node
        final_turn = turns[-1] if turns else None
        if final_turn and final_turn.llm_call and final_turn.llm_call.message:
            tree.root.add(
                "🏁 Final Response",
                data={
                    "type": "final_response",
                    "message": final_turn.llm_call.message,
                    "tooltip": _snippet(final_turn.llm_call.message, 300),
                },
            )

        tree.root.expand()
        for child in tree.root.children:
            child.expand()

        # Select overview node by default
        tree.select_node(overview_node)
        await self.render_detail({"type": "overview", "traj": traj})

    async def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update detail panel whenever tree selection/highlight changes."""
        node_data = event.node.data if event.node else None
        if isinstance(node_data, dict):
            await self.render_detail(node_data)

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle explicit node selection."""
        node_data = event.node.data if event.node else None
        if isinstance(node_data, dict):
            await self.render_detail(node_data)

    async def render_detail(self, data: dict[str, Any]) -> None:
        """Render markdown content in right panel based on selected node.

        Awaiting ``Markdown.update`` keeps its internal gather-future properly
        retrieved, avoiding 'exception was never retrieved' noise when the
        app exits while an update is in flight.
        """
        dtype = data.get("type")
        try:
            md_widget = self.query_one("#detail-markdown", Markdown)
        except NoMatches:
            return  # Widget tree already torn down (e.g. quit with a pending message)

        if dtype == "overview":
            traj: Trajectory = data["traj"]
            md_content = self._format_overview(traj)
        elif dtype == "user_msg":
            md_content = f"# 👤 User Request\n\n```text\n{data.get('text', '')}\n```"
        elif dtype == "turn":
            turn: TrajectoryTurn = data["turn"]
            lc: LlmCall = data["llm"]
            md_content = self._format_turn(turn, lc)
        elif dtype == "tool":
            tc: ToolCall = data["tool"]
            md_content = self._format_tool(tc)
        elif dtype == "skill":
            sl: SkillLoad = data["skill"]
            md_content = self._format_skill(sl)
        elif dtype == "final_response":
            md_content = f"# 🏁 Final Agent Response\n\n{data.get('message', '')}"
        else:
            traj = data.get("traj") or self.current_trajectory
            md_content = self._format_overview(traj) if traj else ""

        await md_widget.update(md_content)

    # ── Detail Formatters ─────────────────────────────────────────────────────

    def _format_overview(self, traj: Trajectory) -> str:
        lines = [
            f"# 🚀 Trajectory Overview: `{traj.profile}`",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Run ID** | `{traj.run_id}` |",
            f"| **Profile** | `{traj.profile}` |",
            f"| **Status** | `{'✓ OK' if traj.status == 'ok' else '✗ ' + str(traj.status)}` |",
            f"| **Started At** | `{traj.started_at}` |",
            f"| **Ended At** | `{traj.ended_at or 'In progress / Unfinished'}` |",
            f"| **LLM Calls** | {len(traj.llm_calls)} |",
            f"| **Tool Invocations** | {len(traj.tool_calls)} |",
            f"| **Skills Loaded** | {len(traj.skill_loads)} |",
            f"| **Prompt Tokens** | {traj.total_prompt_tokens:,} |",
            f"| **Completion Tokens** | {traj.total_completion_tokens:,} |",
            f"| **Total Tokens** | {traj.total_prompt_tokens + traj.total_completion_tokens:,} |",
            "",
            "### 🛠️ Tools Called",
        ]
        if traj.tool_names:
            for t in traj.tool_names:
                count = sum(1 for tc in traj.tool_calls if tc.name == t)
                lines.append(f"- **`{t}`** ({count} call{'s' if count > 1 else ''})")
        else:
            lines.append("*(No tools called)*")

        lines.append("")
        lines.append("### 📚 Skills Loaded")
        if traj.skill_loads:
            for sl in traj.skill_loads:
                lines.append(
                    f"- **`{sl.skill_name}`** (source: `{sl.source or 'default'}`) via `{sl.tool_name or 'agent'}`"
                )
        else:
            lines.append("*(No skill.load marks)*")

        return "\n".join(lines)

    def _format_turn(self, turn: TrajectoryTurn, lc: LlmCall) -> str:
        m_short = short_model_name(lc.model)
        usage = lc.usage or {}
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        t_tok = usage.get("total_tokens", p_tok + c_tok)

        lines = [
            f"# 🔄 Turn {turn.index}: LLM Step",
            "",
            "| Attribute | Value |",
            "|---|---|",
            f"| **Model (Short)** | **`{m_short}`** |",
            f"| **Model (Full)** | `{lc.model or '?'}` |",
            f"| **Prompt Tokens** | {p_tok:,} |",
            f"| **Completion Tokens** | {c_tok:,} |",
            f"| **Total Tokens** | {t_tok:,} |",
            f"| **Started At** | `{lc.started_at}` |",
            f"| **Ended At** | `{lc.ended_at or '—'}` |",
            "",
        ]

        if lc.message:
            lines.extend(
                [
                    "### 💭 Reasoning & Thought Message",
                    "",
                    lc.message,
                    "",
                ]
            )

        if lc.tool_calls:
            lines.extend(
                [
                    "### 🛠️ Requested Tool Calls",
                    "",
                ]
            )
            for i, req in enumerate(lc.tool_calls, 1):
                t_name = req.get("name", "unknown") if isinstance(req, dict) else "?"
                t_args = req.get("arguments", req.get("args", {})) if isinstance(req, dict) else {}
                t_id = req.get("id", "—") if isinstance(req, dict) else "—"
                args_str = json.dumps(t_args, indent=2) if isinstance(t_args, dict) else str(t_args)

                lines.extend(
                    [
                        f"#### {i}. `{t_name}` (ID: `{t_id}`)",
                        "```json",
                        args_str,
                        "```",
                        "",
                    ]
                )

        if turn.tool_calls:
            lines.extend(
                [
                    f"### ⚡ Executed Tools in this Turn ({len(turn.tool_calls)})",
                    "",
                ]
            )
            for tc in turn.tool_calls:
                lines.append(f"- **`{tc.name}`** → *See child tool node for full input/output logs.*")

        return "\n".join(lines)

    def _format_tool(self, tc: ToolCall) -> str:
        lines = [
            f"# 🛠️ Tool Execution: `{tc.name}`",
            "",
            "| Attribute | Value |",
            "|---|---|",
            f"| **Tool Name** | `{tc.name}` |",
            f"| **Tool Call ID** | `{tc.tool_call_id or '—'}` |",
            f"| **Started At** | `{tc.started_at}` |",
            f"| **Ended At** | `{tc.ended_at or '—'}` |",
            "",
            "### 📥 Input Arguments",
            "",
        ]

        # Check if python code or dict
        if tc.name == "python_interpreter" and "code" in tc.args:
            lines.extend(
                [
                    "```python",
                    str(tc.args["code"]).strip(),
                    "```",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "```json",
                    json.dumps(tc.args, indent=2),
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "### 📤 Output Result / Logs",
                "",
            ]
        )

        res = tc.result or ""
        if res.startswith("```"):
            lines.append(res)
        else:
            lines.extend(
                [
                    "```text",
                    res,
                    "```",
                ]
            )

        return "\n".join(lines)

    def _format_skill(self, sl: SkillLoad) -> str:
        return "\n".join(
            [
                f"# 📚 Skill Loaded: `{sl.skill_name}`",
                "",
                "| Attribute | Value |",
                "|---|---|",
                f"| **Skill Name** | `{sl.skill_name}` |",
                f"| **Source** | `{sl.source or 'default'}` |",
                f"| **Trigger Tool** | `{sl.tool_name or 'agent'}` |",
                f"| **Timestamp** | `{sl.timestamp}` |",
            ]
        )
