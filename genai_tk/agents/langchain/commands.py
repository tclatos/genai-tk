"""Unified LangChain agent CLI command (react | deep | custom)."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from loguru import logger
from typer import Option

if TYPE_CHECKING:
    from rich.console import Console

    from genai_tk.agents.langchain.config import AgentProfileConfig


def _get_config_path() -> str:
    """Return the path to the langchain agents config (directory or file).

    Checks for ``{paths.config}/agents/langchain/`` (directory) first,
    then falls back to ``{paths.config}/agents/langchain.yaml``.
    """
    from genai_tk.utils.config_mngr import paths_config

    agents_dir = paths_config().config / "agents"
    dir_path = agents_dir / "langchain"
    if dir_path.is_dir():
        return str(dir_path)
    return str(agents_dir / "langchain.yaml")


def _list_profiles() -> None:
    """Display all agent profiles in a Rich table."""
    from rich.console import Console
    from rich.table import Table

    from genai_tk.agents.langchain.config import load_unified_config

    console = Console()
    try:
        cfg = load_unified_config(_get_config_path())
    except Exception as e:
        _display_config_error(console, e)
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


def _display_config_error(console: "Console", error: Exception) -> None:
    """Display a configuration error as a formatted Rich panel."""
    from rich.panel import Panel
    from rich.text import Text

    from genai_tk.utils.config_exceptions import ConfigError, ConfigValidationError

    if isinstance(error, ConfigValidationError):
        n = len(error.errors)
        count_label = f"{n} validation error{'s' if n != 1 else ''}"
        ctx = f" in {error.config_name}" if error.config_name else ""
        text = Text()
        text.append(f"{count_label}{ctx}\n\n", style="bold red")
        for err_msg in error.errors:
            text.append(f"  • {err_msg}\n", style="red")
        if error.file_path:
            text.append(f"\n📄 File: {error.file_path}\n", style="cyan")
        text.append("\n💡 Check the field value(s) above and update your YAML file.", style="yellow")
        console.print(Panel(text, title="❌  Configuration Validation Error", border_style="red"))
    elif isinstance(error, ConfigError):
        text = Text()
        text.append(f"{error.message}\n", style="red")
        if error.suggestion:
            text.append(f"\n💡 {error.suggestion}", style="yellow")
        console.print(Panel(text, title="❌  Configuration Error", border_style="red"))
    else:
        console.print(f"[red]Error loading agent config:[/red] {error}")


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
        details: Annotated[
            bool,
            Option("--details", help="Show detailed LLM and tool call panels (compact summary by default)"),
        ] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        sandbox: Annotated[
            Optional[str],
            Option("--sandbox", "-b", help="Sandbox override: local (default) | docker"),
        ] = None,
        vnc: Annotated[
            bool,
            Option("--vnc", help="Auto-open VNC in browser for visual debugging (requires --sandbox docker)"),
        ] = False,
        keep_sandbox: Annotated[
            bool,
            Option("--keep-sandbox", help="Keep the sandbox container running after agent exits for inspection"),
        ] = False,
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

        from genai_tk.agents.langchain.agent_cli import run_langchain_agent_direct, run_langchain_agent_shell
        from genai_tk.agents.langchain.langchain_agent import LangchainAgent
        from genai_tk.agents.langchain.setup import setup_langchain

        console = Console()

        # --list
        if list_profiles:
            _list_profiles()
            return

        # Setup LangChain early
        setup_langchain(llm, lc_debug, lc_verbose, cache)

        # Validate agent_type override
        if agent_type and agent_type not in ("react", "deep", "custom"):
            console.print(f"[red]Invalid --type '{agent_type}'. Choose: react | deep | custom[/red]")
            raise typer.Exit(1)

        # Resolve profile name (need config only to get default_profile)
        profile_name: str | None = profile
        if not profile_name:
            from genai_tk.agents.langchain.config import load_unified_config

            try:
                cfg = load_unified_config(_get_config_path())
            except Exception as e:
                _display_config_error(console, e)
                raise typer.Exit(1) from e
            profile_name = cfg.default_profile
            if not profile_name:
                console.print("[red]No profile specified and no default_profile set in langchain.yaml[/red]")
                raise typer.Exit(1)

        # Handle stdin input
        if not input_text and not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()

        # Validate: need input or --chat
        if not chat and (not input_text or len(input_text.strip()) < 1):
            console.print("[red]Error:[/red] Provide a query (positional arg, stdin) or use --chat")
            raise typer.Exit(1)

        try:
            agent = LangchainAgent(
                profile_name,
                llm=llm or None,
                agent_type=agent_type,  # type: ignore[arg-type]
                mcp_servers=list(mcp) if mcp else [],
                checkpointer=chat,
                details=details,
                sandbox=sandbox or None,
                vnc=vnc,
                keep_sandbox=keep_sandbox,
            )
        except ValueError as e:
            config_path = _get_config_path()
            console.print(f"[red]{e}[/red]")
            console.print(f"[dim]Config loaded from: {config_path}[/dim]")
            _list_profiles()
            raise typer.Exit(1) from e

        assert agent._profile is not None
        _display_profile_info(agent._profile, llm or None)

        # Docker sandbox browser tasks typically need multi-turn interaction
        # (e.g. the agent may ask which site to use).  Auto-promote to chat mode
        # so the user can reply, while still sending the initial query.
        effective_chat = chat
        if not chat and sandbox == "docker":
            effective_chat = True
            agent.checkpointer = True
            console.print("[dim]ℹ  --sandbox docker implies --chat (multi-turn). Type /quit to exit.[/dim]")

        try:
            if effective_chat:
                asyncio.run(run_langchain_agent_shell(agent, initial_query=input_text))
            else:
                asyncio.run(run_langchain_agent_direct(input_text, agent, stream=stream))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(0) from None
        except ValueError as e:
            # ValueError with helpful suggestions (e.g., for unknown LLM)
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1) from e
        except Exception as e:
            # Other exceptions - show error and traceback for debugging
            console.print(f"\n[red]Error:[/red] {e}")
            logger.exception("Agent error")
            raise typer.Exit(1) from e
