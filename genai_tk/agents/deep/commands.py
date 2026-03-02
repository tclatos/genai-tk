"""Deep agent CLI helpers and commands."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated, Optional

import typer
from loguru import logger

from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# Deep agent helper functions
# ---------------------------------------------------------------------------


def _list_deep_profiles() -> None:
    """List available deep agent profiles in a Rich table."""
    from rich.console import Console
    from rich.table import Table

    from genai_tk.core.deep_agents import get_default_profile_name, load_deep_agent_profiles

    console = Console()
    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deepagents.yaml")

    try:
        profiles = load_deep_agent_profiles(config_path)
    except Exception as e:
        console.print(f"[red]Error loading profiles:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    if not profiles:
        console.print(f"[yellow]No profiles found in {config_path}[/yellow]")
        return

    default_profile_name = get_default_profile_name()

    table = Table(title=f"🧠 Deep Agent Profiles ({config_path})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Tools", style="green")
    table.add_column("MCP Servers", style="blue")
    table.add_column("Skills", style="yellow")

    for profile in profiles:
        tool_count = len(profile.tool_configs)
        tools_info = f"{tool_count} tool(s)" if tool_count > 0 else "-"

        mcp_servers = ", ".join(profile.mcp_servers) if profile.mcp_servers else "-"

        skills_info = f"{len(profile.skill_directories)} dir(s)" if profile.skill_directories else "-"

        # Mark default profile
        profile_name = profile.name
        if default_profile_name and profile.name == default_profile_name:
            profile_name = f"⭐ {profile_name}"

        table.add_row(
            profile_name,
            profile.description[:50] + "..." if len(profile.description) > 50 else profile.description,
            tools_info,
            mcp_servers,
            skills_info,
        )

    console.print(table)
    if default_profile_name:
        console.print("\n[dim]⭐ = Default profile (used when -p is not specified)[/dim]")


def _display_deep_agent_info(profile_name: str, mcp_servers: list[str], llm_id: str | None) -> None:
    """Display agent configuration info in a Rich panel."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from genai_tk.core.deep_agents import get_deep_agent_profile

    console = Console()

    try:
        profile = get_deep_agent_profile(profile_name)
    except Exception as e:
        console.print(f"[red]Error loading profile:[/red] {e}")
        return

    info = Text()
    info.append("Profile: ", style="bold cyan")
    info.append(f"{profile.name}\n", style="white")

    if profile.description:
        info.append("Description: ", style="bold cyan")
        info.append(f"{profile.description}\n", style="white")

    if llm_id:
        info.append("LLM: ", style="bold cyan")
        info.append(f"{llm_id}\n", style="yellow")
    elif profile.llm:
        info.append("LLM: ", style="bold cyan")
        info.append(f"{profile.llm}\n", style="yellow")

    tool_count = len(profile.tool_configs)
    if tool_count > 0:
        info.append("Tools: ", style="bold cyan")
        info.append(f"{tool_count} configured\n", style="green")

    if profile.mcp_servers or mcp_servers:
        mcp_list = list(set(profile.mcp_servers + mcp_servers))
        info.append("MCP Servers: ", style="bold cyan")
        info.append(f"{', '.join(mcp_list)}\n", style="blue")

    if profile.skill_directories:
        info.append("Skill Directories: ", style="bold cyan")
        info.append(f"{len(profile.skill_directories)} dir(s)\n", style="yellow")

    info.append("Planning: ", style="bold cyan")
    info.append(
        f"{'enabled' if profile.enable_planning else 'disabled'}\n", style="green" if profile.enable_planning else "dim"
    )

    info.append("File System: ", style="bold cyan")
    info.append(
        f"{'enabled' if profile.enable_file_system else 'disabled'}",
        style="green" if profile.enable_file_system else "dim",
    )

    panel = Panel(info, title="🧠 Deep Agent Configuration", border_style="cyan")
    console.print(panel)


async def _run_deep_single_shot(
    profile_name: str,
    input_text: str,
    llm_override: str | None = None,
    extra_mcp_servers: list[str] | None = None,
    stream: bool = False,
) -> None:
    """Run deep agent in single-shot mode."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from genai_tk.core.deep_agents import (
        create_deep_agent_from_profile,
        get_deep_agent_profile,
        run_deep_agent,
    )
    from genai_tk.core.llm_factory import get_llm

    console = Console()

    try:
        profile = get_deep_agent_profile(profile_name)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Resolve LLM
    llm = None
    if llm_override:
        try:
            llm = get_llm(llm_override)
        except Exception as e:
            console.print(f"[red]Error resolving LLM:[/red] {e}", style="bold")
            raise typer.Exit(1) from e

    # Display configuration
    _display_deep_agent_info(profile_name, extra_mcp_servers or [], llm_override)

    # Show user query
    console.print("\n[bold magenta]👤 User Query:[/bold magenta]")
    console.print(f"[italic white]{input_text}[/italic white]\n")

    # Create and run agent
    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        console=console,
    ) as progress:
        task = progress.add_task(f"🧠 {profile_name} agent is thinking...", total=None)

        try:
            # Create agent
            agent = await create_deep_agent_from_profile(
                profile=profile,
                llm=llm,
                extra_mcp_servers=extra_mcp_servers,
            )

            # Run agent
            result = await run_deep_agent(
                agent=agent,
                input_message=input_text,
                stream=stream,
            )

            progress.update(task, description=f"✅ {profile_name} agent completed!")
            progress.update(task, completed=True)

        except Exception:
            progress.update(task, description=f"❌ Error in {profile_name} agent")
            progress.update(task, completed=True)
            raise

    # Display results
    if "messages" in result and result["messages"]:
        response_content = result["messages"][-1].content

        console.print("\n[bold green]🤖 Agent Response:[/bold green]\n")
        try:
            md = Markdown(response_content)
            console.print(md)
        except Exception as e:
            logger.warning(f"Markdown rendering failed: {e}")
            console.print(response_content)


async def _run_deep_chat_mode(
    profile_name: str,
    llm_override: str | None = None,
    extra_mcp_servers: list[str] | None = None,
    stream: bool = False,
) -> None:
    """Run deep agent in interactive chat mode."""
    from langgraph.checkpoint.memory import MemorySaver
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    from rich.console import Console
    from rich.markdown import Markdown

    from genai_tk.core.deep_agents import (
        create_deep_agent_from_profile,
        get_deep_agent_profile,
        run_deep_agent,
    )
    from genai_tk.core.llm_factory import get_llm

    console = Console()

    try:
        profile = get_deep_agent_profile(profile_name)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Resolve LLM
    llm = None
    if llm_override:
        try:
            llm = get_llm(llm_override)
        except Exception as e:
            console.print(f"[red]Error resolving LLM:[/red] {e}", style="bold")
            raise typer.Exit(1) from e

    # Display configuration
    _display_deep_agent_info(profile_name, extra_mcp_servers or [], llm_override)

    # Create agent with checkpointer for memory
    checkpointer = MemorySaver()
    try:
        agent = await create_deep_agent_from_profile(
            profile=profile,
            llm=llm,
            extra_mcp_servers=extra_mcp_servers,
            checkpointer=checkpointer,
        )
    except Exception as e:
        console.print(f"\n[red]Error creating agent:[/red] {e}", style="bold")
        logger.exception("Agent creation failed")
        raise typer.Exit(1) from e

    # Create prompt session
    console.print("\n[bold cyan]🧠 Deep Agent Chat Mode[/bold cyan]")
    console.print("[dim]Commands: /quit, /clear, /help[/dim]\n")

    prompt_style = Style.from_dict({"prompt": "cyan bold"})
    session = PromptSession(style=prompt_style)

    thread_id = "chat-session"

    while True:
        try:
            # Use async prompt to avoid event loop conflict
            user_input = await session.prompt_async("You: ")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.strip() == "/quit":
                console.print("[yellow]Exiting chat mode...[/yellow]")
                break
            elif user_input.strip() == "/clear":
                console.clear()
                continue
            elif user_input.strip() == "/help":
                console.print("\n[bold]Available commands:[/bold]")
                console.print("  /quit  - Exit chat mode")
                console.print("  /clear - Clear screen")
                console.print("  /help  - Show this help\n")
                continue

            # Run agent
            try:
                result = await run_deep_agent(
                    agent=agent,
                    input_message=user_input,
                    thread_id=thread_id,
                    stream=stream,
                )

                if "messages" in result and result["messages"]:
                    response_content = result["messages"][-1].content

                    console.print("\n[bold green]Agent:[/bold green]")
                    try:
                        md = Markdown(response_content)
                        console.print(md)
                    except Exception:
                        console.print(response_content)
                    console.print()

            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n", style="bold")
                logger.exception("Chat error")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
            continue
        except EOFError:
            break


# ---------------------------------------------------------------------------
# CLI command registration
# ---------------------------------------------------------------------------


def register(cli_app: typer.Typer) -> None:
    """Register the ``deep`` command with *cli_app*."""

    @cli_app.command("deep")
    def deep(
        input_text: Annotated[
            Optional[str],
            typer.Argument(help="User query text (positional). If omitted, reads from stdin or uses --chat mode."),
        ] = None,
        profile: Annotated[
            Optional[str],
            typer.Option(
                "--profile",
                "-p",
                help="Profile name from deepagents.yaml (required unless --list)",
            ),
        ] = None,
        chat: Annotated[
            bool,
            typer.Option(
                "--chat",
                "-c",
                help="Interactive multi-turn chat mode (REPL). Use /quit to exit.",
            ),
        ] = False,
        llm: Annotated[
            Optional[str],
            typer.Option("--llm", "-m", help="LLM identifier (ID or tag) to override profile default"),
        ] = None,
        mcp: Annotated[
            Optional[list[str]],
            typer.Option("--mcp", help="Additional MCP server names (merged with profile's servers)"),
        ] = None,
        stream: Annotated[bool, typer.Option("--stream", "-s", help="Stream intermediate agent steps")] = False,
        list_profiles: Annotated[
            bool, typer.Option("--list", "-l", help="List available deep agent profiles and exit")
        ] = False,
    ) -> None:
        """Run a Deep Agent with planning, file system, and subagent capabilities.

        Deep Agents use deepagents v0.4+ for complex, multi-step tasks with built-in planning tools,
        file system operations, and subagent spawning.

        Examples:

          # List available profiles
          cli agents deep --list

          # Run with a profile (single-shot)
          cli agents deep -p Research "Latest developments in quantum computing"

          # Interactive chat mode
          cli agents deep -p Coding --chat

          # Override LLM and add MCP servers
          cli agents deep -p Research -m gpt_4@openai --mcp tavily-mcp "Research AI trends"

          # Read from stdin
          echo "Debug this code" | cli agents deep -p Coding
        """
        from rich.console import Console

        from genai_tk.core.deep_agents import get_default_profile_name

        console = Console()

        # Handle --list flag
        if list_profiles:
            _list_deep_profiles()
            return

        # Handle stdin input
        if not input_text and not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()

        # Get default profile if not specified
        if not profile:
            profile = get_default_profile_name()
            console.print(f"[dim]Using default profile: {profile}[/dim]")

        # Validate input (not required for chat mode)
        if not chat and (not input_text or len(input_text) < 3):
            console.print("[red]Error:[/red] Input text required (or use --chat for interactive mode)", style="bold")
            raise typer.Exit(1)

        # Run the agent
        try:
            if chat:
                # Interactive chat mode
                asyncio.run(_run_deep_chat_mode(profile, llm, mcp, stream))
            else:
                # Single-shot mode - input_text guaranteed to be valid here by validation above
                if not input_text:
                    console.print("[red]Error:[/red] No input provided", style="bold")
                    raise typer.Exit(1)
                asyncio.run(_run_deep_single_shot(profile, input_text, llm, mcp, stream))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            raise typer.Exit(0) from None
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}", style="bold")
            logger.exception("Deep agent error")
            raise typer.Exit(1) from e
