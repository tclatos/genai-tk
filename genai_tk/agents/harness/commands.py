"""Unified CLI commands that work across both LangChain and DeerFlow agent profiles.

Provides a harness-agnostic entry point (``cli agents run`` / ``cli agents list``)
on top of :func:`genai_tk.agents.harness.create_harness`. Framework-specific
commands (``cli agents langchain``, ``cli agents deerflow``) remain available for
flags that only make sense for one harness (sandbox, mode, --chat REPL, etc.).
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from typer import Option


def register(cli_app: typer.Typer) -> None:
    """Register the ``run`` and ``list`` commands with *cli_app*."""

    @cli_app.command("run")
    def run(
        profile: Annotated[str, typer.Argument(help="Profile key (LangChain) or profile name (DeerFlow)")],
        query: Annotated[str, typer.Argument(help="Query text")],
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier override")] = None,
        thread_id: Annotated[Optional[str], Option("--thread-id", "-t", help="Conversation thread ID")] = None,
        json_output: Annotated[bool, Option("--json", help="Print raw NDJSON events instead of rendered text")] = False,
    ) -> None:
        """Run any LangChain or DeerFlow agent profile through the unified harness layer.

        Resolves PROFILE across both config trees automatically and streams the
        response, regardless of which harness backs it.

        Examples:
            uv run cli agents run research "Summarize recent AI safety news"
            uv run cli agents run "Web Browser" "Go to atos.net" --llm gpt_41mini@openai
        """
        import asyncio

        from genai_tk.agents.harness import (
            EndEvent,
            ErrorEvent,
            TokenEvent,
            ToolCallEvent,
            ToolResultEvent,
            create_harness,
        )

        async def _run() -> None:
            harness = create_harness(profile, llm_override=llm)
            try:
                async for event in harness.astream(query, thread_id=thread_id):
                    if json_output:
                        typer.echo(event.model_dump_json())
                        continue
                    if isinstance(event, TokenEvent):
                        typer.echo(event.text, nl=False)
                    elif isinstance(event, ToolCallEvent):
                        typer.secho(f"\n[tool] {event.tool_name}({event.args})", fg=typer.colors.CYAN)
                    elif isinstance(event, ToolResultEvent):
                        typer.secho(f"[tool result] {event.content[:200]}", fg=typer.colors.CYAN)
                    elif isinstance(event, ErrorEvent):
                        typer.secho(f"\n[error] {event.message}", fg=typer.colors.RED)
                    elif isinstance(event, EndEvent):
                        typer.echo()
            finally:
                await harness.aclose()

        try:
            asyncio.run(_run())
        except ValueError as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED)
            raise typer.Exit(1) from e

    @cli_app.command("list")
    def list_profiles() -> None:
        """List all agent profiles across both LangChain and DeerFlow config trees."""
        from rich.console import Console
        from rich.table import Table

        from genai_tk.agents.harness import list_harness_profiles

        console = Console()
        table = Table(title="🤖 All Agent Profiles (LangChain + DeerFlow)")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Harness", style="yellow", no_wrap=True)
        table.add_column("LLM", style="magenta")
        table.add_column("Description", style="white")
        for ref in list_harness_profiles():
            table.add_row(ref.key, ref.harness, ref.llm or "(default)", ref.description)
        console.print(table)
        console.print(
            "[dim]Use 'cli agents run KEY \"query\"', "
            "or 'cli agents langchain'/'cli agents deerflow' for framework-specific flags.[/dim]"
        )
