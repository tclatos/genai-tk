"""Unified CLI commands that work across every agent harness.

``cli agents run`` and ``cli agents list`` are the single entry points for
running and inspecting agent profiles. ``run`` resolves a profile key against
the unified ``agents:`` dict via :func:`genai_tk.agents.harness.create_harness`
and streams the canonical event model — it works for LangChain (react | deep |
custom) and DeerFlow profiles alike. Framework-specific behaviour is exposed
through a small set of cross-harness flags (``--chat``, ``--mode``,
``--sandbox``, ``--mcp``) rather than separate subcommands.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from typer import Option

console = Console()


def _resolve_default_profile() -> str:
    """Return the configured default profile key, or raise typer.Exit."""
    from genai_tk.agents.harness.profiles import load_agent_profiles

    try:
        _profiles, _defaults, default_key = load_agent_profiles()
    except Exception as e:
        console.print(f"[red]Error loading agent config:[/red] {e}")
        raise typer.Exit(1) from e
    if not default_key:
        console.print("[red]No profile specified and no default_profile set in agent_defaults[/red]")
        raise typer.Exit(1)
    return default_key


def register(cli_app: typer.Typer) -> None:
    """Register the ``run`` and ``list`` commands with *cli_app*."""

    @cli_app.command("run")
    def run(
        profile: Annotated[
            Optional[str],
            typer.Argument(help="Profile key/slug or name (omit to use agent_defaults.default_profile)."),
        ] = None,
        query: Annotated[
            Optional[str],
            typer.Argument(help="Query text. Omit to read from stdin or use --chat."),
        ] = None,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier override")] = None,
        chat: Annotated[
            bool, Option("--chat", "-c", help="Interactive multi-turn chat REPL. Use /quit to exit.")
        ] = False,
        mode: Annotated[
            Optional[str],
            Option("--mode", help="DeerFlow reasoning mode override: flash | thinking | pro | ultra"),
        ] = None,
        sandbox: Annotated[
            Optional[str],
            Option("--sandbox", "-b", help="DeerFlow sandbox override: local | docker"),
        ] = None,
        mcp: Annotated[
            list[str],
            Option("--mcp", help="Additional MCP server names appended to the profile (repeatable)"),
        ] = [],
        thread_id: Annotated[Optional[str], Option("--thread-id", "-t", help="Conversation thread ID")] = None,
        json_output: Annotated[bool, Option("--json", help="Print raw NDJSON events instead of rendered text")] = False,
        trace: Annotated[bool, Option("--trace", help="Show node-level execution trace lines")] = False,
        verbose: Annotated[bool, Option("--verbose", "-v", help="Enable DEBUG logging")] = False,
    ) -> None:
        """Run any LangChain or DeerFlow agent profile through the unified harness layer.

        Resolves PROFILE across the unified ``agents:`` config automatically and
        streams the response, regardless of which harness backs it.

        Examples:
            uv run cli agents run research "Summarize recent AI safety news"
            uv run cli agents run "Web Browser" "Go to atos.net" --llm gpt_41mini@openai
            uv run cli agents run research --chat
            uv run cli agents run "Research Assistant" --mode ultra --sandbox docker
            echo "What is RAG?" | uv run cli agents run research
        """
        import sys

        if verbose:
            from loguru import logger

            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        from genai_tk.agents.harness import create_harness

        profile_key = profile or _resolve_default_profile()
        if not profile:
            console.print(f"[dim]Using default profile: {profile_key}[/dim]")

        # stdin input when no positional query and not a TTY
        if not query and not chat and not sys.stdin.isatty():
            query = sys.stdin.read().strip()

        if not chat and (not query or not query.strip()):
            console.print("[red]Error:[/red] Provide a query (positional arg, stdin) or use --chat")
            raise typer.Exit(1)

        try:
            harness = create_harness(
                profile_key,
                llm_override=llm,
                force_memory_checkpointer=chat,
                mode_override=mode,
                sandbox_override=sandbox,
                extra_mcp=list(mcp) if mcp else None,
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        import asyncio

        async def _main() -> None:
            try:
                if chat:
                    from genai_tk.agents.harness.chat_repl import run_chat_repl

                    await run_chat_repl(harness, initial_query=query, show_trace=trace, console=console)
                else:
                    from genai_tk.agents.harness.chat_repl import astream_turn

                    assert query is not None  # guaranteed by the `not chat` check above
                    await astream_turn(
                        harness,
                        query,
                        thread_id=thread_id,
                        show_trace=trace,
                        json_output=json_output,
                        console=console,
                    )
            finally:
                await harness.aclose()

        try:
            # A single asyncio.run() drives harness creation-adjacent work, the
            # turn(s), and aclose() on ONE event loop. Splitting these across
            # separate asyncio.run() calls breaks backends (e.g. AioSandboxBackend)
            # that bind async resources (HTTP clients, subprocess transports) to
            # the loop active when the backend started — a later aclose() on a
            # different loop then fails to actually kill the sandbox container,
            # silently leaking it (see genai_tk/agents/sandbox/aio_backend.py).
            asyncio.run(_main())
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(0) from None

    @cli_app.command("tui")
    def tui(
        profile: Annotated[
            Optional[str],
            typer.Argument(help="DeerFlow profile key/name (omit to use deerflow.default_profile)."),
        ] = None,
        message: Annotated[
            Optional[str],
            typer.Argument(help="Optional initial prompt sent when the TUI opens."),
        ] = None,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier override")] = None,
        mode: Annotated[
            Optional[str],
            Option("--mode", help="DeerFlow reasoning mode override: flash | thinking | pro | ultra"),
        ] = None,
        sandbox: Annotated[
            Optional[str],
            Option("--sandbox", "-b", help="DeerFlow sandbox override: local | docker"),
        ] = None,
        mcp: Annotated[
            list[str],
            Option("--mcp", help="Additional MCP server names appended to the profile (repeatable)"),
        ] = [],
        resume: Annotated[Optional[str], Option("--resume", help="Resume a thread by id or title")] = None,
        continue_recent: Annotated[bool, Option("--continue", help="Resume the most recent thread in the TUI")] = False,
        verbose: Annotated[bool, Option("--verbose", "-v", help="Enable DEBUG logging")] = False,
    ) -> None:
        """Launch the DeerFlow terminal workbench (TUI) for a profile.

        The TUI is the interactive sibling of ``cli agents run --chat``: a
        Textual app over the embedded DeerFlow client with a transcript, status
        line, slash-command palette, and thread switching. Only DeerFlow
        profiles are supported today; other harnesses can plug in a TUI later.

        Examples:
            uv run cli agents tui simple-deerflow
            uv run cli agents tui "Research Assistant" --mode ultra
            uv run cli agents tui research --resume my-thread
            uv run cli agents tui simple-deerflow -- "What is RAG?"
        """
        import sys

        if verbose:
            from loguru import logger

            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        from genai_tk.agents.deer_flow.profile import DeerFlowError
        from genai_tk.agents.deer_flow.runtime import get_default_profile_name
        from genai_tk.agents.deer_flow.tui import run_deerflow_tui
        from genai_tk.agents.harness.registry import lookup_profile

        profile_key = profile or get_default_profile_name()
        if not profile_key:
            console.print("[red]No profile specified and no deerflow.default_profile set[/red]")
            raise typer.Exit(1)
        if not profile:
            console.print(f"[dim]Using default profile: {profile_key}[/dim]")

        try:
            resolved = lookup_profile(profile_key)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        if resolved.harness != "deerflow":
            console.print(
                f"[yellow]TUI not yet supported for harness '{resolved.harness}'[/yellow]. "
                f"Use [cyan]cli agents run {profile_key} --chat[/cyan] instead."
            )
            raise typer.Exit(1)

        try:
            run_deerflow_tui(
                resolved.name,
                llm_override=llm,
                mode_override=mode,
                sandbox_override=sandbox,
                extra_mcp=list(mcp) if mcp else None,
                message=message,
                thread_id=resume,
                continue_recent=continue_recent,
                verbose=verbose,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(0) from None
        except (ValueError, ImportError, DeerFlowError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

    @cli_app.command("list")
    def list_profiles() -> None:
        """List all agent profiles across every harness."""
        from genai_tk.agents.harness import list_harness_profiles

        table = Table(title="🤖 Agent Profiles")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Kind", style="yellow", no_wrap=True)
        table.add_column("Harness", style="magenta", no_wrap=True)
        table.add_column("LLM", style="green")
        table.add_column("Description", style="white")
        for ref in list_harness_profiles():
            table.add_row(ref.key, ref.kind, ref.harness, ref.llm or "(default)", ref.description)
        console.print(table)
        console.print("[dim]Use 'cli agents run KEY \"query\"' (add --chat for a REPL).[/dim]")
