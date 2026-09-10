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
from textual.widgets import Footer, Header, Markdown, Select, Static, Tree

from genai_tk.utils.trajectory_store import (
    LlmCall,
    SkillLoad,
    ToolCall,
    Trajectory,
    TrajectoryStore,
    TrajectoryTurn,
    short_model_name,
)


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
                yield Tree("Execution Timeline", id="timeline-tree")
            with VerticalScroll(id="right-panel"):
                yield Markdown("", id="detail-markdown")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the view once widgets are mounted."""
        if self.runs:
            target_id = self.initial_run_id or self.runs[0].run_id
            self.load_run(target_id)
        else:
            self.query_one("#detail-markdown", Markdown).update(
                "# No Trajectories Found\n\nNo recorded runs were found in the trajectory store."
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle run dropdown selection changes."""
        if event.value and event.value != Select.BLANK:
            self.load_run(str(event.value))

    def action_toggle_focus(self) -> None:
        """Toggle focus between tree and detail panels."""
        tree = self.query_one("#timeline-tree", Tree)
        details = self.query_one("#right-panel", VerticalScroll)
        if tree.has_focus:
            details.focus()
        else:
            tree.focus()

    def action_reload_run(self) -> None:
        """Reload the current run from disk."""
        if self.current_trajectory:
            self.load_run(self.current_trajectory.run_id)

    def load_run(self, run_id: str) -> None:
        """Load and populate a trajectory into the UI."""
        traj = self.store.get(run_id)
        if traj is None:
            self.query_one("#detail-markdown", Markdown).update(
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
        tree.root.data = {"type": "root", "traj": traj}

        # 1. Overview Node
        overview_node = tree.root.add("📋 Run Overview", data={"type": "overview", "traj": traj})

        # 2. User Prompt Node
        user_msg = self.store._root_user_message(traj)
        if user_msg:
            tree.root.add("👤 User Request", data={"type": "user_msg", "text": user_msg})

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
                turn_node = tree.root.add(turn_label, data={"type": "turn", "turn": turn, "llm": lc})

                for tc in turn.tool_calls:
                    tool_label = f"🛠️ [magenta]tool[/] [bold]{tc.name}[/]"
                    turn_node.add(tool_label, data={"type": "tool", "tool": tc})

                for sl in turn.skill_loads:
                    skill_label = f"📚 [yellow]skill.load[/] {sl.skill_name}"
                    turn_node.add(skill_label, data={"type": "skill", "skill": sl})
            else:
                for tc in turn.tool_calls:
                    tree.root.add(f"🛠️ [magenta]tool[/] [bold]{tc.name}[/]", data={"type": "tool", "tool": tc})
                for sl in turn.skill_loads:
                    tree.root.add(f"📚 [yellow]skill.load[/] {sl.skill_name}", data={"type": "skill", "skill": sl})

        # 4. Final Response Node
        final_turn = turns[-1] if turns else None
        if final_turn and final_turn.llm_call and final_turn.llm_call.message:
            tree.root.add(
                "🏁 Final Response",
                data={"type": "final_response", "message": final_turn.llm_call.message},
            )

        # 5. Raw Events Node
        tree.root.add("🔍 Raw Events (ATOF)", data={"type": "raw_events", "traj": traj})

        tree.root.expand()
        for child in tree.root.children:
            child.expand()

        # Select overview node by default
        tree.select_node(overview_node)
        self.render_detail({"type": "overview", "traj": traj})

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update detail panel whenever tree selection/highlight changes."""
        node_data = event.node.data if event.node else None
        if isinstance(node_data, dict):
            self.render_detail(node_data)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle explicit node selection."""
        node_data = event.node.data if event.node else None
        if isinstance(node_data, dict):
            self.render_detail(node_data)

    def render_detail(self, data: dict[str, Any]) -> None:
        """Render markdown content in right panel based on selected node."""
        dtype = data.get("type")
        md_widget = self.query_one("#detail-markdown", Markdown)

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
        elif dtype == "raw_events":
            traj: Trajectory = data["traj"]
            md_content = self._format_raw_events(traj)
        else:
            traj = data.get("traj") or self.current_trajectory
            md_content = self._format_overview(traj) if traj else ""

        md_widget.update(md_content)

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

    def _format_raw_events(self, traj: Trajectory) -> str:
        lines = [
            f"# 🔍 Raw ATOF Events ({len(traj.events)} total)",
            "",
            "```json",
            json.dumps(traj.events, indent=2),
            "```",
        ]
        return "\n".join(lines)
