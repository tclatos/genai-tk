"""CLI-facing agent runners: interactive shell and single-shot direct execution.

These functions handle Rich console rendering and user interaction.
They receive a ``LangchainAgent`` and delegate heavy-lifting to it.
"""

from __future__ import annotations

import time
import webbrowser
from pathlib import Path

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from genai_tk.agents.langchain.langchain_agent import LangchainAgent, _extract_content
from genai_tk.agents.rich_display import ASSISTANT_BORDER_STYLE, ASSISTANT_PANEL_TITLE
from genai_tk.utils.markdown import looks_like_markdown


def _render_content(content: str, console: Console, *, elapsed: float | None = None) -> None:
    """Render assistant response using Rich, auto-detecting Markdown."""
    time_suffix = f"  [dim]({elapsed:.1f}s)[/dim]" if elapsed is not None else ""
    if not content or not content.strip():
        console.print(
            Panel(
                "[bold yellow]⚠ Agent returned an empty response.\n"
                "Possible causes: model timeout, empty LLM output, or tool error.\n"
                "Use /logs to inspect the trace or try with --debug.[/bold yellow]",
                title=f"[bold white on dark_orange3] Assistant [/bold white on dark_orange3]{time_suffix}",
                border_style="dark_orange3",
            )
        )
        logger.warning("Agent returned empty response")
        return
    if looks_like_markdown(content):
        body: Markdown | str = Markdown(content)
    else:
        body = content
    console.print(
        Panel(
            body,
            title=f"{ASSISTANT_PANEL_TITLE}{time_suffix}",
            border_style=ASSISTANT_BORDER_STYLE,
        )
    )


async def run_langchain_agent_shell(agent: LangchainAgent, initial_query: str | None = None) -> None:
    """Run an interactive multi-turn shell for a ``LangchainAgent``.

    Initialises the agent once, then loops over prompts. Type ``/quit`` to exit.

    Args:
        agent: A configured ``LangchainAgent``. ``checkpointer=True`` is
            recommended for conversation memory.
        initial_query: If provided, execute this query before entering the
            interactive loop (useful for ``--sandbox docker`` auto-chat).
    """
    console = Console()
    assert agent._profile is not None

    _setup_log_buffer()

    # Write a persistent trace log for post-mortem analysis
    log_path = Path("tmp/agent_trace.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_sink_id = logger.add(
        str(log_path),
        level="DEBUG",
        rotation="2 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}",
    )

    welcome_text = Text(f"🤖 {agent._profile.name} Agent  [{agent._profile.type}]", style="bold cyan")
    if agent._profile.mcp_servers:
        welcome_text.append(f"\nMCP servers: {', '.join(agent._profile.mcp_servers)}", style="green")
    welcome_text.append(f"\nTrace log: {log_path.resolve()}", style="dim")
    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace, /logs\nUse up/down arrows to navigate prompt history[/dim]\n")

    with console.status("[bold green]Initialising agent...[/bold green]"):
        compiled = await agent._ensure_initialized()

    history_file = Path(".blueprint.input.history")
    session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

    # Execute the initial query before entering the interactive loop
    if initial_query and initial_query.strip():
        console.print(Panel(initial_query, title="[bold blue]User[/bold blue]", border_style="blue"))
        try:
            t0 = time.monotonic()
            result = await compiled.ainvoke(
                {"messages": initial_query},
                {"configurable": {"thread_id": "1"}},
            )
            elapsed = time.monotonic() - t0
            _render_content(_extract_content(result), console, elapsed=elapsed)
            console.print()
        except Exception as e:
            console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
            logger.exception("Agent error during initial query")

    try:
        while True:
            try:
                with patch_stdout():
                    prompt_style = Style.from_dict({"prompt": "bold cyan"})
                    user_input = await session.prompt_async(
                        ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                    )

                user_input = user_input.strip()
                if user_input.lower() in ("/quit", "/exit", "/q"):
                    console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                    break
                if user_input == "/help":
                    console.print(
                        Panel(
                            "/help   – show this help\n"
                            "/quit   – exit the shell\n"
                            "/trace  – open last LangSmith trace in browser\n"
                            "/logs   – show recent log entries (LLM calls, tools, errors)",
                            title="[bold cyan]Commands[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    continue
                if user_input == "/trace":
                    webbrowser.open("https://smith.langchain.com/")
                    continue
                if user_input == "/logs":
                    _show_recent_logs(console)
                    continue
                if user_input.startswith("/"):
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                    continue
                if not user_input:
                    continue

                console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

                t0 = time.monotonic()
                result = await compiled.ainvoke(
                    {"messages": user_input},
                    {"configurable": {"thread_id": "1"}},
                )
                elapsed = time.monotonic() - t0
                _render_content(_extract_content(result), console, elapsed=elapsed)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
                break
            except Exception as e:  # pragma: no cover
                console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
                logger.exception("Agent error")
    finally:
        logger.remove(log_sink_id)
        await agent.close()


# ---------------------------------------------------------------------------
# Log buffer for /logs command
# ---------------------------------------------------------------------------

_LOG_BUFFER: list[str] = []
_LOG_SINK_ID: int | None = None
_MAX_LOG_ENTRIES = 200


def _setup_log_buffer() -> None:
    """Add a loguru sink that captures messages into an in-memory buffer."""
    global _LOG_SINK_ID  # noqa: PLW0603
    if _LOG_SINK_ID is not None:
        return

    def _sink(message: str) -> None:
        _LOG_BUFFER.append(str(message).rstrip())
        if len(_LOG_BUFFER) > _MAX_LOG_ENTRIES:
            del _LOG_BUFFER[: len(_LOG_BUFFER) - _MAX_LOG_ENTRIES]

    _LOG_SINK_ID = logger.add(
        _sink, level="DEBUG", format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}"
    )


def _show_recent_logs(console: Console, last_n: int = 50) -> None:
    """Display the most recent log entries."""
    entries = _LOG_BUFFER[-last_n:]
    if not entries:
        console.print("[dim]No log entries captured yet.[/dim]")
        return
    console.print(
        Panel(
            "\n".join(entries),
            title=f"[bold cyan]Recent Logs ({len(entries)} entries)[/bold cyan]",
            border_style="cyan",
        )
    )


async def run_langchain_agent_direct(
    query: LanguageModelInput,
    agent: LangchainAgent,
    stream: bool = False,
) -> None:
    """Execute a single query using a ``LangchainAgent`` and render with Rich.

    Args:
        query: The user query to execute.
        agent: A configured ``LangchainAgent``.
        stream: If ``True``, stream intermediate steps (deep agents only).
    """
    console = Console()

    with console.status("[bold green]Initialising agent...[/bold green]"):
        compiled = await agent._ensure_initialized()

    assert agent._profile is not None
    system_prompt = agent._profile.system_prompt or agent._profile.pre_prompt

    messages: list[HumanMessage] = []
    if system_prompt:
        messages.append(HumanMessage(content=system_prompt))
    messages.append(HumanMessage(content=query))

    if stream:
        async for chunk in compiled.astream({"messages": messages}):
            content = _extract_content(chunk)
            if content:
                _render_content(content, console)
    else:
        result = await compiled.ainvoke({"messages": messages})
        _render_content(_extract_content(result), console)

    await agent.close()
