"""CLI-facing agent runners: interactive shell and single-shot direct execution.

These functions handle Rich console rendering and user interaction.
They receive a ``LangchainAgent`` and delegate heavy-lifting to it.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
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
from genai_tk.utils.markdown import looks_like_markdown


def _render_content(content: str, console: Console) -> None:
    """Render assistant response using Rich, auto-detecting Markdown."""
    if looks_like_markdown(content):
        body: Markdown | str = Markdown(content)
    else:
        body = content
    console.print(
        Panel(
            body,
            title="[bold white on royal_blue1] Assistant [/bold white on royal_blue1]",
            border_style="royal_blue1",
        )
    )


async def run_langchain_agent_shell(agent: LangchainAgent) -> None:
    """Run an interactive multi-turn shell for a ``LangchainAgent``.

    Initialises the agent once, then loops over prompts. Type ``/quit`` to exit.

    Args:
        agent: A configured ``LangchainAgent``. ``checkpointer=True`` is
            recommended for conversation memory.
    """
    console = Console()
    assert agent._profile is not None

    welcome_text = Text(f"🤖 {agent._profile.name} Agent  [{agent._profile.type}]", style="bold cyan")
    if agent._profile.mcp_servers:
        welcome_text.append(f"\nMCP servers: {', '.join(agent._profile.mcp_servers)}", style="green")
    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace\nUse up/down arrows to navigate prompt history[/dim]\n")

    with console.status("[bold green]Initialising agent...[/bold green]"):
        compiled = await agent._ensure_initialized()

    history_file = Path(".blueprint.input.history")
    session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

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
                            "/trace  – open last LangSmith trace in browser",
                            title="[bold cyan]Commands[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    continue
                if user_input == "/trace":
                    webbrowser.open("https://smith.langchain.com/")
                    continue
                if user_input.startswith("/"):
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                    continue
                if not user_input:
                    continue

                console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

                result = await compiled.ainvoke(
                    {"messages": user_input},
                    {"configurable": {"thread_id": "1"}},
                )
                _render_content(_extract_content(result), console)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
                break
            except Exception as e:  # pragma: no cover
                console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
    finally:
        await agent.close()


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
