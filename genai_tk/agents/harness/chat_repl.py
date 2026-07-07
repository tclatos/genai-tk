"""Harness-driven terminal streaming renderer and chat REPL.

A single :func:`stream_turn` renders canonical harness events to a Rich console
(used by both ``cli agents run`` single-shot and the ``--chat`` REPL), and
:func:`run_chat_repl` wraps it in a ``prompt_toolkit`` loop with slash commands.
Both work for any :class:`~genai_tk.agents.harness.base.BaseHarness` — this is
the one shared terminal interaction model for every agent runtime.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from genai_tk.agents.harness import (
    ClarificationEvent,
    EndEvent,
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from genai_tk.agents.harness.base import BaseHarness

_DEFAULT_CONSOLE = Console()


async def _astream_turn(
    harness: BaseHarness,
    query: str,
    *,
    thread_id: str | None = None,
    show_trace: bool = False,
    json_output: bool = False,
    console: Console | None = None,
) -> str:
    """Run one turn against *harness*, render events, return the assistant text.

    Args:
        harness: Any ready-to-stream harness session.
        query: The user message.
        thread_id: Conversation thread ID.
        show_trace: Print node-level trace lines.
        json_output: Print one JSON event per line and skip rendering.
        console: Rich console to render to (default: a fresh one).

    Returns:
        The concatenated assistant token text for the turn.
    """
    console = console or _DEFAULT_CONSOLE
    parts: list[str] = []

    async for event in harness.astream(query, thread_id=thread_id):
        if json_output:
            console.print(event.model_dump_json())
            continue
        if isinstance(event, TokenEvent):
            console.print(event.text, end="", highlight=False)
            parts.append(event.text)
        elif isinstance(event, ToolCallEvent):
            console.print(f"\n[tool] {event.tool_name}({event.args})", style="cyan")
        elif isinstance(event, ToolResultEvent):
            content = (event.content or "")[:200]
            console.print(f"[tool result] {content}", style="cyan")
        elif isinstance(event, NodeEvent):
            if show_trace:
                console.print(f"→ {event.node}", style="dim")
        elif isinstance(event, ClarificationEvent):
            console.print(f"\n❓ {event.question}", style="yellow")
            parts.append(event.question)
        elif isinstance(event, ErrorEvent):
            console.print(f"\n[error] {event.message}", style="red")
        elif isinstance(event, EndEvent):
            pass
    return "".join(parts)


def stream_turn(
    harness: BaseHarness,
    query: str,
    *,
    thread_id: str | None = None,
    show_trace: bool = False,
    json_output: bool = False,
    console: Console | None = None,
) -> str:
    """Synchronous wrapper around :func:`_astream_turn` for single-shot use."""
    return asyncio.run(
        _astream_turn(
            harness,
            query,
            thread_id=thread_id,
            show_trace=show_trace,
            json_output=json_output,
            console=console,
        )
    )


def _print_info(harness: BaseHarness, console: Console) -> None:
    """Render a small panel with the harness/profile/model for ``/info``."""
    name = getattr(harness, "name", "?")
    profile = getattr(harness, "profile", None)
    model = getattr(harness, "model_name", None)
    lines: list[str] = [f"Harness: {name}"]
    if profile is not None:
        lines.append(f"Profile: {getattr(profile, 'name', '?')}")
    if model:
        lines.append(f"Model:   {model}")
    console.print(Panel("\n".join(lines), title="[bold cyan]Agent[/bold cyan]", border_style="cyan"))


async def run_chat_repl(
    harness: BaseHarness,
    *,
    initial_query: str | None = None,
    show_trace: bool = False,
    console: Console | None = None,
) -> None:
    """Interactive multi-turn REPL over *harness*.

    Uses ``prompt_toolkit`` with history + auto-suggest. Slash commands:
    ``/help``, ``/info``, ``/clear`` (new thread), ``/quit``.

    Args:
        harness: Any harness session. Eagerly ``ensure_ready()``-ied when the
            adapter supports it so ``/info`` can show the resolved profile/model.
        initial_query: Optional first message sent before entering the loop.
        show_trace: Print node-level trace lines for each turn.
        console: Rich console to render to (default: a fresh one).
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style

    console = console or _DEFAULT_CONSOLE
    # Eagerly prepare adapters that expose ensure_ready() (DeerFlow) so /info
    # can report the resolved profile/model before the first turn.
    ensure_ready = getattr(harness, "ensure_ready", None)
    if callable(ensure_ready):
        try:
            await ensure_ready()
        except Exception as exc:
            console.print(f"[red]Failed to prepare agent: {exc}[/red]")
            return

    thread_id = uuid.uuid4().hex
    console.print(Panel.fit("Agent Chat", style="bold cyan"))
    _print_info(harness, console)
    console.print("[dim]Commands: /help  /info  /clear  /quit[/dim]")
    console.print()

    session: PromptSession = PromptSession(history=FileHistory(str(Path(".agents.input.history"))))
    prompt_style = Style.from_dict({"prompt": "bold green"})

    async def _turn(user_input: str) -> None:
        console.print(Panel(user_input, title="[bold blue]You[/bold blue]", border_style="blue"))
        await _astream_turn(harness, user_input, thread_id=thread_id, show_trace=show_trace, console=console)
        console.print()

    if initial_query:
        await _turn(initial_query)

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async(
                    ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                )
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        elif cmd == "/clear":
            thread_id = uuid.uuid4().hex
            console.print("[yellow]New conversation thread started.[/yellow]")
            continue
        elif cmd == "/help":
            console.print(
                Panel(
                    "/help   show this help\n/info   show current agent\n/clear  start a fresh conversation thread\n/quit   exit",
                    title="[cyan]Chat Commands[/cyan]",
                    border_style="cyan",
                )
            )
            continue
        elif cmd == "/info":
            _print_info(harness, console)
            continue
        elif user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input!r}[/yellow]  Type /help for available commands.")
            continue

        try:
            await _turn(user_input)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Use /quit to exit.[/yellow]")
        except Exception as exc:
            console.print(Panel(f"[red]Error: {exc}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
