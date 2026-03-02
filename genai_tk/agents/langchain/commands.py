"""Unified LangChain agent CLI command (react | deep | custom)."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from loguru import logger
from typer import Option

from genai_tk.utils.config_mngr import global_config

if TYPE_CHECKING:
    from genai_tk.agents.langchain.config import AgentProfileConfig


def _get_config_path() -> str:
    """Return the path to the unified langchain.yaml config file."""
    config_dir = global_config().get_dir_path("paths.config")
    return str(config_dir / "agents" / "langchain.yaml")


def _list_profiles() -> None:
    """Display all agent profiles in a Rich table."""
    from rich.console import Console
    from rich.table import Table

    from genai_tk.agents.langchain.config import load_unified_config

    console = Console()
    try:
        cfg = load_unified_config(_get_config_path())
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from e

    table = Table(title=f"🤖 LangChain Agent Profiles  (default: {cfg.default_profile!r})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow", no_wrap=True)
    table.add_column("LLM", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Tools", style="green")
    table.add_column("MCP Servers", style="blue")
    table.add_column("Features", style="dim")

    for p in cfg.profiles:
        is_default = p.name == cfg.default_profile
        name_cell = f"⭐ {p.name}" if is_default else p.name
        tools_cell = f"{len(p.tools)}" if p.tools else "-"
        mcp_cell = ", ".join(p.mcp_servers) if p.mcp_servers else "-"
        desc = p.description[:50] + "…" if len(p.description) > 50 else p.description
        features = ", ".join(p.features[:2]) if p.features else "-"
        table.add_row(name_cell, p.type, p.llm or "(default)", desc, tools_cell, mcp_cell, features)

    console.print(table)
    console.print("[dim]⭐ = default profile  |  use --profile NAME or -p NAME to select[/dim]")


def _display_profile_info(profile: "AgentProfileConfig", llm_override: str | None) -> None:
    """Show selected profile configuration as a Rich panel."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    info = Text()
    info.append("Profile: ", style="bold cyan")
    info.append(f"{profile.name}\n", style="white")
    info.append("Type: ", style="bold cyan")
    info.append(f"{profile.type}\n", style="yellow")
    if llm_override:
        info.append("LLM: ", style="bold cyan")
        info.append(f"{llm_override} (override)\n", style="yellow")
    elif profile.llm:
        info.append("LLM: ", style="bold cyan")
        info.append(f"{profile.llm}\n", style="yellow")
    if profile.mcp_servers:
        info.append("MCP Servers: ", style="bold cyan")
        info.append(f"{', '.join(profile.mcp_servers)}\n", style="blue")
    if profile.features:
        info.append("Features: ", style="bold cyan")
        info.append(f"{', '.join(profile.features)}\n", style="dim")

    console.print(Panel(info, title="🤖 Agent Configuration", border_style="cyan"))


def register(cli_app: typer.Typer) -> None:
    """Register the ``langchain`` command with *cli_app*."""

    @cli_app.command("langchain")
    def langchain_cmd(
        input_text: Annotated[
            Optional[str],
            typer.Argument(help="Query text. Omit to read from stdin or use --chat."),
        ] = None,
        profile: Annotated[
            Optional[str],
            Option(
                "--profile", "-p", help="Profile name from langchain.yaml (default: langchain_agents.default_profile)"
            ),
        ] = None,
        agent_type: Annotated[
            Optional[str],
            Option("--type", "-t", help="Override agent engine: react | deep | custom"),
        ] = None,
        llm: Annotated[
            Optional[str],
            Option("--llm", "-m", help="LLM identifier (ID or tag) overriding profile default"),
        ] = None,
        chat: Annotated[
            bool,
            Option("--chat", "-c", help="Interactive multi-turn chat mode. Use /quit to exit."),
        ] = False,
        mcp: Annotated[
            list[str],
            Option("--mcp", help="Additional MCP server names (appended to profile servers)"),
        ] = [],
        stream: Annotated[
            bool,
            Option("--stream", "-s", help="Stream intermediate steps (deep agents)"),
        ] = False,
        list_profiles: Annotated[
            bool,
            Option("--list", "-l", help="List available agent profiles and exit"),
        ] = False,
        cache: Annotated[
            str,
            Option("--cache", help="Cache strategy: sqlite | memory | no_cache"),
        ] = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
    ) -> None:
        """Run a LangChain-based agent (react | deep | custom) from a YAML profile.

        Examples:

          # List all configured profiles
          cli agents langchain --list

          # Single-shot with default profile (Research / deep agent)
          cli agents langchain "Latest trends in quantum computing"

          # Select a lightweight react profile
          cli agents langchain -p filesystem "List Python files"

          # Interactive chat with a deep agent
          cli agents langchain -p Coding --chat

          # Override engine and LLM at runtime
          cli agents langchain -p Research --type react --llm gpt_41mini@openai "Quick summary of AI news"

          # Read query from stdin
          echo "Research renewable energy" | cli agents langchain -p Research

          # Add extra MCP servers
          cli agents langchain -p filesystem --mcp playwright "Browse atos.net"
        """
        from rich.console import Console

        from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
        from genai_tk.agents.langchain.setup import setup_langchain
        from genai_tk.core.llm_factory import LlmFactory

        console = Console()

        # --list
        if list_profiles:
            _list_profiles()
            return

        # Setup LangChain early
        setup_langchain(llm, lc_debug, lc_verbose, cache)

        # Load config
        try:
            cfg = load_unified_config(_get_config_path())
        except Exception as e:
            console.print(f"[red]Error loading agent config:[/red] {e}")
            raise typer.Exit(1) from e

        # Resolve profile name
        profile_name = profile or cfg.default_profile
        if not profile_name:
            console.print("[red]No profile specified and no default_profile set in langchain.yaml[/red]")
            raise typer.Exit(1)

        # Validate agent_type override
        if agent_type and agent_type not in ("react", "deep", "custom"):
            console.print(f"[red]Invalid --type '{agent_type}'. Choose: react | deep | custom[/red]")
            raise typer.Exit(1)

        # Resolve profile (merges defaults, warns about incompatibilities)
        try:
            resolved = resolve_profile(cfg, profile_name, type_override=agent_type)  # type: ignore[arg-type]
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            _list_profiles()
            raise typer.Exit(1) from e

        # Resolve and validate LLM override
        llm_override: str | None = None
        if llm:
            try:
                llm_override = LlmFactory.resolve_llm_identifier(llm)
            except ValueError as e:
                console.print(f"[red]Invalid LLM identifier:[/red] {e}")
                raise typer.Exit(1) from e

        # Display configuration info
        _display_profile_info(resolved, llm_override)

        # Handle stdin input
        if not input_text and not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()

        # Validate: need input or --chat
        if not chat and (not input_text or len(input_text.strip()) < 1):
            console.print("[red]Error:[/red] Provide a query (positional arg, stdin) or use --chat")
            raise typer.Exit(1)

        from genai_tk.agents.langchain.agent import (
            run_langchain_agent_direct,
            run_langchain_agent_shell,
        )

        try:
            if chat:
                asyncio.run(
                    run_langchain_agent_shell(
                        resolved,
                        llm_override=llm_override,
                        extra_mcp_servers=mcp or None,
                    )
                )
            else:
                asyncio.run(
                    run_langchain_agent_direct(
                        input_text,
                        resolved,
                        llm_override=llm_override,
                        extra_mcp_servers=mcp or None,
                        stream=stream,
                    )
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(0) from None
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            logger.exception("Agent error")
            raise typer.Exit(1) from e
