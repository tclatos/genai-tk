"""Rich terminal renderer for Agent-User Interaction Protocol (AUIP) events."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

from genai_tk.agents.rich_display import summarize_tool_result
from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipErrorEvent,
    AuipEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipThinkingEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
)


class RichAuipRenderer:
    """Renders an AUIP event stream to a Rich console with live status and clean formatting."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_trace: bool = False,
        raw: bool = False,
    ) -> None:
        self.console = console or Console()
        self.show_trace = show_trace
        self.raw = raw
        self.status = None
        self.parts: list[str] = []
        self._planning_shown = False

    def render_event(self, event: AuipEvent) -> None:
        """Render a single AUIP event to the terminal."""
        if isinstance(event, AuipTokenEvent):
            if self.raw:
                self.console.print(event.text, end="", highlight=False)
            else:
                if self.status is None and self.console.is_terminal:
                    self.status = self.console.status("[dim]Generating response...[/dim]")
                    self.status.start()
            self.parts.append(event.text)

        elif isinstance(event, AuipThinkingEvent):
            self._stop_status()
            if self.show_trace:
                self.console.print(event.text, end="", style="dim italic", highlight=False)

        elif isinstance(event, AuipPhaseEvent):
            self._stop_status()
            # If phase event contains planning todos
            todos = event.state.get("todos")
            if todos and isinstance(todos, list):
                self._render_planning(todos)
            elif self.show_trace and event.phase:
                self.console.print(f"→ {event.phase}", style="dim")

        elif isinstance(event, AuipToolCallEvent):
            self._stop_status()
            args_str = str(event.args) if event.args else ""
            self.console.print(f"\n[cyan]\\[tool][/cyan] [yellow]{event.tool_name}[/yellow]({args_str})")

        elif isinstance(event, AuipToolResultEvent):
            self._stop_status()
            brief = summarize_tool_result(event.tool_name, event.content or "")
            dur_str = f" [dim]({event.duration_ms:.0f}ms)[/dim]" if event.duration_ms > 0 else ""
            status_mark = "✓" if event.status == "success" else "✗"
            color = "cyan" if event.status == "success" else "red"
            self.console.print(f"[{color}]\\[tool result] {status_mark} {brief}{dur_str}[/{color}]")

        elif isinstance(event, AuipArtifactEvent):
            self._stop_status()
            if event.artifact_type == "skill":
                self.console.print(f"  [dim]📖 Loaded skill:[/dim] [yellow]{event.title}[/yellow]")
            else:
                self.console.print(
                    Panel(
                        event.content[:2000] + ("…" if len(event.content) > 2000 else ""),
                        title=f"[bold green]Artifact: {event.title}[/bold green]",
                        border_style="green",
                        padding=(0, 1),
                    )
                )

        elif isinstance(event, AuipHitlRequestEvent):
            self._stop_status()
            self.console.print(f"\n❓ [yellow]{event.question}[/yellow]")
            self.parts.append(event.question)

        elif isinstance(event, AuipUsageTelemetryEvent):
            if self.show_trace and (event.input_tokens or event.output_tokens):
                self.console.print(
                    f"[dim]Tokens: {event.input_tokens} in / {event.output_tokens} out | Model: {event.model_name}[/dim]"
                )

        elif isinstance(event, AuipErrorEvent):
            self._stop_status()
            self.console.print(f"\n[red]\\[error] {event.message}[/red]")

        elif isinstance(event, AuipEndEvent):
            self._stop_status()

    def _render_planning(self, todos: list[dict[str, Any]]) -> None:
        header = (
            "[bold cyan]📋 Plan update[/bold cyan]" if self._planning_shown else "[bold cyan]📋 Planning[/bold cyan]"
        )
        self._planning_shown = True
        lines = [header]
        for todo in todos:
            text = todo.get("content", str(todo))
            status = todo.get("status", "pending")
            if status == "completed":
                lines.append(f"  [dim]✓ {text}[/dim]")
            elif status == "in_progress":
                lines.append(f"  [bold]▶ {text}[/bold]")
            else:
                lines.append(f"  □ {text}")
        self.console.print("\n".join(lines))

    def _stop_status(self) -> None:
        if self.status:
            self.status.stop()
            self.status = None

    def finalize(self) -> str:
        """Clean up status and return accumulated response text."""
        self._stop_status()
        return "".join(self.parts)
