"""CLI commands for Deer-flow agents.

Provides command-line interface for running Deer-flow agents with profile-based
configuration, chat mode, streaming, and LLM/MCP customization.
"""

import asyncio
import sys
import uuid
import webbrowser
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from genai_tk.core.llm_factory import LlmFactory, get_llm
from genai_tk.main.cli import CliTopCommand

console = Console()


def _display_agent_tools_and_skills(profile: Any, console: Console) -> None:
    """Display loaded tools and skills information.

    Args:
        profile: Agent profile configuration
        console: Rich console for output
    """
    from omegaconf import OmegaConf
    from rich.panel import Panel
    from rich.text import Text

    from genai_tk.extra.agents.deer_flow.config_bridge import load_skills_from_directories
    from genai_tk.utils.config_mngr import global_config

    info = Text()

    # Display sandbox mode
    config = global_config().root
    sandbox_provider = OmegaConf.select(config, "deerflow.sandbox.provider", default="local")
    if sandbox_provider == "docker":
        info.append("🐳 Sandbox: ", style="bold cyan")
        info.append("Docker (full file I/O, bash, code execution)\n", style="bold green")
    else:
        info.append("💻 Sandbox: ", style="bold cyan")
        info.append("Local (limited file I/O — set sandbox.provider: docker for full support)\n", style="yellow")
    info.append("\n")

    # Display tools information
    if profile.tool_groups:
        info.append("🔧 Tool Groups: ", style="bold cyan")
        info.append(f"{', '.join(profile.tool_groups)}\n", style="white")
        info.append("   Note: Actual tools loaded depend on available dependencies\n\n", style="dim")

    if profile.tool_configs:
        info.append("📦 Extra Tools: ", style="bold cyan")
        info.append(f"{len(profile.tool_configs)} configured\n\n", style="white")

    # Display skills information
    if profile.skill_directories:
        try:
            skills = load_skills_from_directories(profile.skill_directories)
            if skills:
                info.append("🎯 Available Skills: ", style="bold cyan")
                info.append(f"{len(skills)}\n", style="white")
                # Show all skills, grouped by lines of ~80 chars
                skill_names = [s.split("/")[-1] for s in skills]
                line = "   "
                for i, name in enumerate(skill_names):
                    if len(line) + len(name) + 2 > 80:  # Wrap at ~80 chars
                        info.append(line, style="green")
                        info.append("\n", style="white")
                        line = "   " + name
                    else:
                        line += name
                    if i < len(skill_names) - 1:
                        line += ", "
                if line.strip():
                    info.append(line, style="green")
                info.append("\n   Note: Skills add sub-workflows for complex tasks\n", style="dim")
        except Exception as e:
            logger.debug(f"Could not load skills info: {e}")

    if info.plain:
        console.print(Panel(info, title="[bold cyan]Agent Configuration[/bold cyan]", border_style="cyan"))


def _get_llm_display_name(llm: Any, llm_id: str | None = None) -> str:
    """Get display name for LLM in format model_id@provider.

    Args:
        llm: The LLM instance
        llm_id: Optional explicit LLM ID (e.g., 'gpt_41mini@openai')

    Returns:
        Display string like 'gpt_41mini@openai' or model name as fallback
    """
    # If llm_id is explicitly provided, use it
    if llm_id:
        return llm_id

    # Try to get the llm_id if it's a LlmFactory instance
    if hasattr(llm, "llm_id") and llm.llm_id:
        return llm.llm_id

    # Fallback to model_name or model attribute
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")

    # Try to get provider
    provider = getattr(llm, "provider", None)
    if provider:
        return f"{model_name}@{provider}"

    return model_name


def _display_output_files(thread_id: str, thread_data_middleware: Any, console: Console) -> None:
    """Display output files created by the agent with their real disk paths.

    Args:
        thread_id: The conversation thread ID
        thread_data_middleware: ConfigThreadDataMiddleware instance (or None)
        console: Rich console for output
    """
    if thread_data_middleware is None:
        return

    from genai_tk.extra.agents.deer_flow.config_thread_data_middleware import ConfigThreadDataMiddleware

    if not isinstance(thread_data_middleware, ConfigThreadDataMiddleware):
        return

    paths = thread_data_middleware._get_thread_paths(thread_id)
    from pathlib import Path

    from rich.panel import Panel
    from rich.text import Text

    output_files = []
    for dir_key in ("outputs_path", "workspace_path"):
        dir_path = Path(paths[dir_key])
        if dir_path.exists():
            for f in dir_path.rglob("*"):
                if f.is_file():
                    output_files.append(f)

    if not output_files:
        return

    info = Text()
    for f in sorted(output_files, key=lambda p: p.stat().st_mtime, reverse=True):
        size_kb = f.stat().st_size / 1024
        info.append(f"  📄 {f.name}", style="bold green")
        info.append(f"  ({size_kb:.1f} KB)\n", style="dim")
        info.append(f"     {f}\n", style="dim cyan")

    console.print(Panel(info, title="[bold green]📁 Output Files[/bold green]", border_style="green"))


async def _process_message(
    user_input: str,
    agent: Any,
    thread_id: str,
    stream_enabled: bool,
    console: Console,
    trace_enabled: bool = True,
    thread_data_middleware: Any = None,
) -> None:
    """Process a single message through the agent and display the response.

    Args:
        user_input: The user's message
        agent: The deer-flow agent
        thread_id: The conversation thread ID
        stream_enabled: Whether to stream intermediate steps
        console: Rich console for output
        trace_enabled: Whether Rich trace middleware is active (disables duplicate display)
        thread_data_middleware: Optional middleware for resolving output file paths
    """
    from genai_tk.extra.agents.deer_flow.agent import run_deer_flow_agent

    # Only display user input if trace middleware is not active (to avoid duplication)
    if not trace_enabled:
        console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

    # Execute agent
    if stream_enabled:
        current_node = None

        def on_node(node: str) -> None:
            nonlocal current_node
            if node != current_node:
                current_node = node
                console.print(f"[dim]→ {node}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🦌 Agent thinking...", total=None)
            response = await run_deer_flow_agent(
                agent=agent,
                user_input=user_input,
                thread_id=thread_id,
                on_node=on_node,
            )
            progress.update(task, description="✅ Complete")
    else:
        with console.status("🦌 Agent thinking...", spinner="dots"):
            response = await run_deer_flow_agent(
                agent=agent,
                user_input=user_input,
                thread_id=thread_id,
            )

    # Only display response if trace middleware is not active (to avoid duplication)
    if not trace_enabled:
        console.print()
        console.print(
            Panel(
                Markdown(response),
                title="[bold white on royal_blue1] Assistant [/bold white on royal_blue1]",
                border_style="royal_blue1",
            )
        )

    # Show output files created by the agent (resolve virtual paths to real disk paths)
    _display_output_files(thread_id, thread_data_middleware, console)


class DeerFlowCommands(CliTopCommand, BaseModel):
    """CLI commands for Deer-flow agents."""

    description: str = "Deer-flow agent commands for interactive AI with advanced reasoning"

    def get_description(self) -> tuple[str, str]:
        """Return command name and description for CLI registration."""
        return "deerflow", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        """Register Deer-flow CLI subcommands."""

        @cli_app.callback(invoke_without_command=True)
        def main(
            ctx: typer.Context,
            input_text: Annotated[
                Optional[str],
                typer.Argument(help="User query text (positional). If omitted, reads from stdin or uses --chat mode."),
            ] = None,
            profile: Annotated[
                Optional[str],
                typer.Option(
                    "--profile",
                    "-p",
                    help="Profile name from deerflow.yaml (required unless --list)",
                ),
            ] = None,
            chat: Annotated[
                bool,
                typer.Option(
                    "--chat",
                    "-s",
                    help="Interactive multi-turn chat mode (REPL). Use /quit to exit.",
                ),
            ] = False,
            llm: Annotated[
                Optional[str],
                typer.Option(
                    "--llm",
                    "-m",
                    help="LLM model override (e.g., 'gpt_41_openrouter', 'ollama/llama3.2')",
                ),
            ] = None,
            mcp: Annotated[
                list[str],
                typer.Option(
                    "--mcp",
                    help="Additional MCP server names to enable (merged with profile's list)",
                ),
            ] = [],
            mode: Annotated[
                Optional[str],
                typer.Option(
                    "--mode",
                    help="Override agent mode: flash|thinking|pro|ultra",
                ),
            ] = None,
            stream: Annotated[
                bool,
                typer.Option(
                    "--stream",
                    help="Stream intermediate agent steps in real-time",
                ),
            ] = False,
            list_profiles: Annotated[
                bool,
                typer.Option(
                    "--list",
                    help="List available profiles from deerflow.yaml and exit",
                ),
            ] = False,
            verbose: Annotated[
                bool,
                typer.Option(
                    "--verbose",
                    "-v",
                    help="Enable verbose logging (DEBUG level) for detailed tracing",
                ),
            ] = False,
        ) -> None:
            """Run Deer-flow agents with advanced reasoning capabilities.

            Note: Due to CLI runtime limitations, some deer-flow middlewares are automatically disabled
            in both modes (ThreadDataMiddleware, UploadsMiddleware, TitleMiddleware, MemoryMiddleware,
            plus ClarificationMiddleware in single-shot). Skills requiring file operations may not work
            as expected. For full functionality, use deer-flow natively with its web interface.

            Examples:
                # List available profiles
                cli deerflow --list

                # Run with a profile
                cli deerflow -p "Research Assistant" "Explain quantum computing"

                # Interactive chat mode (recommended for skills like ppt-generation)
                cli deerflow -p "Research Assistant" --chat

                # With LLM override
                cli deerflow -p "Coder" --llm gpt_41_openrouter "Write a sorting algorithm"

                # Stream intermediate steps
                cli deerflow -p "Research Assistant" --stream "What are AI trends?"

                # Enable verbose logging for debugging
                cli deerflow -p "Research Assistant" --verbose "Complex query"

                # Add extra MCP servers
                cli deerflow -p "Coder" --mcp math --mcp weather "Calculate weather patterns"

                # Override mode
                cli deerflow -p "Web Browser" --mode ultra "Research topic"

                # Read from stdin
                echo "What is RAG?" | cli deerflow -p "Research Assistant"
            """
            # Handle --list flag
            if list_profiles:
                _list_profiles()
                return

            # Use default profile if not specified
            if not profile:
                default_profile = _get_default_profile_name()
                if default_profile:
                    profile = default_profile
                    console.print(f"[dim]Using default profile: {profile}[/dim]")
                else:
                    console.print(
                        "[red]Error:[/red] --profile/-p is required (or use --list to see available profiles)",
                        style="bold",
                    )
                    raise typer.Exit(1)

            # Get input from stdin if not provided
            if not input_text and not chat:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read().strip()
                    if not input_text:
                        console.print("[red]Error:[/red] No input provided", style="bold")
                        raise typer.Exit(1)
                else:
                    console.print(
                        "[yellow]Warning:[/yellow] No input provided. Use positional argument, stdin, or --chat mode.",
                        style="bold",
                    )
                    raise typer.Exit(1)

            # Run agent
            try:
                if chat:
                    asyncio.run(
                        _run_chat_mode(
                            profile_name=profile,
                            llm_override=llm,
                            extra_mcp=mcp,
                            mode_override=mode,
                            stream_enabled=stream,
                            initial_input=input_text,
                            verbose=verbose,
                        )
                    )
                else:
                    asyncio.run(
                        _run_single_shot(
                            profile_name=profile,
                            user_input=input_text,
                            llm_override=llm,
                            extra_mcp=mcp,
                            mode_override=mode,
                            stream_enabled=stream,
                            verbose=verbose,
                        )
                    )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                raise typer.Exit(0) from None
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}", style="bold")
                logger.exception("Deer-flow agent error")
                raise typer.Exit(1) from e


def _list_profiles() -> None:
    """List available Deer-flow profiles in a Rich table."""
    from genai_tk.extra.agents.deer_flow.agent import load_deer_flow_profiles
    from genai_tk.utils.config_mngr import global_config

    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deerflow.yaml")

    try:
        profiles = load_deer_flow_profiles(config_path)
    except Exception as e:
        console.print(f"[red]Error loading profiles:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    if not profiles:
        console.print(f"[yellow]No profiles found in {config_path}[/yellow]")
        return

    # Try to load default profile name from config
    default_profile_name = _get_default_profile_name()

    table = Table(title=f"🦌 Deer-flow Profiles ({config_path})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Mode", style="magenta")
    table.add_column("Tool Groups", style="green")
    table.add_column("MCP Servers", style="blue")
    table.add_column("Skill Directories", style="yellow")

    for profile in profiles:
        tool_groups = ", ".join(profile.tool_groups) if profile.tool_groups else "-"
        mcp_servers = ", ".join(profile.mcp_servers) if profile.mcp_servers else "-"

        # Show skill_directories if available, otherwise show legacy skills
        if profile.skill_directories:
            skills_info = "\n".join(profile.skill_directories)
        elif profile.skills:
            skills_info = f"Legacy: {', '.join(profile.skills)}"
        else:
            skills_info = "-"

        # Mark default profile
        profile_name = profile.name
        if default_profile_name and profile.name == default_profile_name:
            profile_name = f"⭐ {profile_name}"

        table.add_row(
            profile_name,
            profile.mode or "flash",
            tool_groups,
            mcp_servers,
            skills_info,
        )

    console.print(table)
    if default_profile_name:
        console.print("\n[dim]⭐ = Default profile (used when -p is not specified)[/dim]")


def _get_default_profile_name() -> str | None:
    """Get the default profile name from global config.

    Returns:
        Default profile name or None if not configured
    """
    from genai_tk.utils.config_mngr import global_config

    try:
        return global_config().get("deerflow.default_profile")
    except Exception:
        return None


def _display_agent_info(profile_dict: Any, llm: Any, llm_id: str | None = None) -> None:
    """Display comprehensive agent configuration information.

    Args:
        profile_dict: The DeerFlowAgentConfig profile
        llm: The LLM instance being used
        llm_id: The resolved LLM identifier (e.g., 'gpt_41mini@openai')
    """
    from genai_tk.extra.agents.deer_flow.config_bridge import load_skills_from_directories

    # Build info sections
    info_lines = []

    # Profile info
    info_lines.append("[bold cyan]Profile Information:[/bold cyan]")
    info_lines.append(f"  Name: {profile_dict.name}")
    info_lines.append(f"  Description: {profile_dict.description}")
    info_lines.append(f"  Mode: {profile_dict.mode}")
    if profile_dict.llm:
        info_lines.append(f"  Configured LLM: {profile_dict.llm}")
    info_lines.append("")

    # LLM info
    llm_display = _get_llm_display_name(llm, llm_id)
    info_lines.append("[bold cyan]Language Model:[/bold cyan]")
    info_lines.append(f"  LLM ID: {llm_display}")
    # Additional details if available
    model_name = getattr(llm, "model", None)
    if model_name and model_name != llm_display:
        info_lines.append(f"  Model Name: {model_name}")
    info_lines.append("")

    # Tool groups
    info_lines.append("[bold cyan]Tool Groups:[/bold cyan]")
    if profile_dict.tool_groups:
        for tg in profile_dict.tool_groups:
            info_lines.append(f"  • {tg}")
    else:
        info_lines.append("  (none)")
    info_lines.append("")

    # MCP Servers
    info_lines.append("[bold cyan]MCP Servers:[/bold cyan]")
    if profile_dict.mcp_servers:
        for mcp in profile_dict.mcp_servers:
            info_lines.append(f"  • {mcp}")
    else:
        info_lines.append("  (none)")
    info_lines.append("")

    # Skills
    info_lines.append("[bold cyan]Skills:[/bold cyan]")
    if profile_dict.skill_directories:
        info_lines.append("  Directories:")
        for skill_dir in profile_dict.skill_directories:
            info_lines.append(f"    • {skill_dir}")
        # Try to load and count skills
        try:
            discovered_skills = load_skills_from_directories(profile_dict.skill_directories)
            info_lines.append(f"  Discovered: {len(discovered_skills)} skills")
            # Show first few
            if discovered_skills:
                info_lines.append("  Sample:")
                for skill in discovered_skills[:5]:
                    info_lines.append(f"    • {skill}")
                if len(discovered_skills) > 5:
                    info_lines.append(f"    ... and {len(discovered_skills) - 5} more")
        except Exception as e:
            info_lines.append(f"  [yellow]Warning: Could not load skills: {e}[/yellow]")
    elif profile_dict.skills:
        info_lines.append("  Legacy mode:")
        for skill in profile_dict.skills:
            info_lines.append(f"    • {skill}")
    else:
        info_lines.append("  (none)")
    info_lines.append("")

    # Additional tools
    if profile_dict.tool_configs:
        info_lines.append("[bold cyan]Additional Tools:[/bold cyan]")
        for tool_cfg in profile_dict.tool_configs:
            if "factory" in tool_cfg:
                info_lines.append(f"  • {tool_cfg['factory']}")
        info_lines.append("")

    # Features
    if profile_dict.features:
        info_lines.append("[bold cyan]Features:[/bold cyan]")
        for feature in profile_dict.features:
            info_lines.append(f"  {feature}")
        info_lines.append("")

    # Display as panel
    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold white on royal_blue1] Agent Configuration [/bold white on royal_blue1]",
            border_style="royal_blue1",
        )
    )


async def _run_single_shot(
    profile_name: str,
    user_input: str,
    llm_override: Optional[str],
    extra_mcp: list[str],
    mode_override: Optional[str],
    stream_enabled: bool,
    verbose: bool = False,
) -> None:
    """Execute a single query (non-interactive mode).

    Args:
        verbose: Enable verbose logging for detailed tracing
    """
    from genai_tk.extra.agents.deer_flow.agent import (
        DeerFlowError,
        create_deer_flow_agent_simple,
        load_deer_flow_profiles,
        run_deer_flow_agent,
        validate_mcp_servers,
        validate_mode,
        validate_profile_name,
    )
    from genai_tk.utils.config_mngr import global_config

    # Configure verbose logging if requested
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )
        logger.debug("Verbose logging enabled in _run_single_shot")

    # Load and validate profile
    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deerflow.yaml")

    try:
        profiles = load_deer_flow_profiles(config_path)
        profile_dict = validate_profile_name(profile_name, profiles)
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Apply overrides with validation
    try:
        if mode_override:
            validated_mode = validate_mode(mode_override)
            profile_dict.mode = validated_mode

        if extra_mcp:
            validated_mcp = validate_mcp_servers(extra_mcp)
            profile_dict.mcp_servers = list(set(profile_dict.mcp_servers + validated_mcp))
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Get LLM
    resolved_llm_id = None
    if llm_override:
        try:
            # Resolve identifier with helpful error messages
            llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm_override)
            if error_msg:
                console.print(error_msg)
                raise typer.Exit(1)

            # Create LLM instance
            llm = get_llm(llm=llm_id)
            resolved_llm_id = llm_id
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Error creating LLM '{llm_override}':[/red] {e}", style="bold")
            console.print(
                "\n[yellow]💡 Tip:[/yellow] Check that the model exists and your API key is configured.\n"
                "Use [cyan]uv run cli info models[/cyan] to see available models."
            )
            raise typer.Exit(1) from e
    elif profile_dict.llm:
        try:
            # Use LLM from profile configuration
            llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(profile_dict.llm)
            if error_msg:
                console.print(error_msg)
                raise typer.Exit(1)

            llm = get_llm(llm=llm_id)
            resolved_llm_id = llm_id
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Error creating LLM from profile '{profile_dict.llm}':[/red] {e}", style="bold")
            console.print(
                "\n[yellow]💡 Tip:[/yellow] Check that the model exists in your profile configuration.\n"
                "Use [cyan]uv run cli info models[/cyan] to see available models."
            )
            raise typer.Exit(1) from e
    else:
        llm = get_llm()

    # Show configuration
    llm_display = _get_llm_display_name(llm, resolved_llm_id)
    console.print(f"[cyan]Profile:[/cyan] {profile_dict.name}")
    console.print(f"[cyan]Mode:[/cyan] {profile_dict.mode}")
    console.print(f"[cyan]LLM:[/cyan] {llm_display}")
    if profile_dict.mcp_servers:
        console.print(f"[cyan]MCP Servers:[/cyan] {', '.join(profile_dict.mcp_servers)}")
    console.print()

    # Create agent (reused across turns)
    with console.status("🦌 Setting up Deer-flow agent...", spinner="dots"):
        from genai_tk.extra.agents.deer_flow.config_thread_data_middleware import ConfigThreadDataMiddleware

        checkpointer = MemorySaver()
        thread_data_mw = ConfigThreadDataMiddleware()
        agent = create_deer_flow_agent_simple(
            profile=profile_dict,
            llm=llm,
            checkpointer=checkpointer,
            thread_data_middleware=thread_data_mw,
            interactive_mode=False,  # Non-interactive mode - no ClarificationMiddleware
        )

    thread_id = str(uuid.uuid4())

    # Execute with streaming or non-streaming
    if stream_enabled:
        current_node = None

        def on_node(node: str) -> None:
            nonlocal current_node
            if node != current_node:
                current_node = node
                console.print(f"[dim]→ {node}[/dim]")

        def on_content(node: str, content: str) -> None:
            # We'll just rely on the final result for now
            pass

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🦌 Agent thinking...", total=None)
            response = await run_deer_flow_agent(
                agent=agent,
                user_input=user_input,
                thread_id=thread_id,
                on_node=on_node,
                on_content=on_content,
            )
            progress.update(task, description="✅ Complete")
    else:
        with console.status("🦌 Agent thinking...", spinner="dots"):
            response = await run_deer_flow_agent(
                agent=agent,
                user_input=user_input,
                thread_id=thread_id,
            )

    # Display result
    console.print()
    panel = Panel(
        Markdown(response),
        title="🦌 Deer-flow Response",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)

    # Show output files with real disk paths
    _display_output_files(thread_id, thread_data_mw, console)


async def _run_chat_mode(
    profile_name: str,
    llm_override: Optional[str],
    extra_mcp: list[str],
    mode_override: Optional[str],
    stream_enabled: bool,
    initial_input: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Run interactive chat REPL mode.

    Args:
        initial_input: Optional first message to process before entering interactive mode.
        verbose: Enable verbose logging for detailed tracing
    """
    from genai_tk.extra.agents.deer_flow.agent import (
        DeerFlowError,
        create_deer_flow_agent_simple,
        load_deer_flow_profiles,
        validate_mcp_servers,
        validate_mode,
        validate_profile_name,
    )
    from genai_tk.utils.config_mngr import global_config

    # Configure verbose logging if requested
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )
        logger.debug("Verbose logging enabled in _run_chat_mode")

    # Load and validate profile
    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deerflow.yaml")

    try:
        profiles = load_deer_flow_profiles(config_path)
        profile_dict = validate_profile_name(profile_name, profiles)
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Apply overrides with validation
    try:
        if mode_override:
            validated_mode = validate_mode(mode_override)
            profile_dict.mode = validated_mode

        if extra_mcp:
            validated_mcp = validate_mcp_servers(extra_mcp)
            profile_dict.mcp_servers = list(set(profile_dict.mcp_servers + validated_mcp))
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    # Get LLM (priority: override > profile > default)
    resolved_llm_id = None
    if llm_override:
        try:
            # Resolve identifier with helpful error messages
            llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm_override)
            if error_msg:
                console.print(error_msg)
                raise typer.Exit(1)

            # Create LLM instance
            llm = get_llm(llm=llm_id)
            resolved_llm_id = llm_id
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Error creating LLM '{llm_override}':[/red] {e}", style="bold")
            console.print(
                "\n[yellow]💡 Tip:[/yellow] Check that the model exists and your API key is configured.\n"
                "Use [cyan]uv run cli info models[/cyan] to see available models."
            )
            raise typer.Exit(1) from e
    elif profile_dict.llm:
        try:
            # Use LLM from profile configuration
            llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(profile_dict.llm)
            if error_msg:
                console.print(error_msg)
                raise typer.Exit(1)

            llm = get_llm(llm=llm_id)
            resolved_llm_id = llm_id
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Error creating LLM from profile '{profile_dict.llm}':[/red] {e}", style="bold")
            console.print(
                "\n[yellow]💡 Tip:[/yellow] Check that the model exists in your profile configuration.\n"
                "Use [cyan]uv run cli info models[/cyan] to see available models."
            )
            raise typer.Exit(1) from e
    else:
        llm = get_llm()

    # Show configuration
    llm_display = _get_llm_display_name(llm, resolved_llm_id)
    console.print(Panel.fit("🦌 Deer-flow Interactive Chat", style="bold cyan"))
    console.print(f"[cyan]Profile:[/cyan] {profile_dict.name}")
    console.print(f"[cyan]Mode:[/cyan] {profile_dict.mode}")
    console.print(f"[cyan]LLM:[/cyan] {llm_display}")
    if profile_dict.mcp_servers:
        console.print(f"[cyan]MCP Servers:[/cyan] {', '.join(profile_dict.mcp_servers)}")
    console.print()
    console.print("[dim]Commands: /quit, /exit, /clear, /help, /info, /trace[/dim]")
    console.print("[dim]Use up/down arrows to navigate prompt history[/dim]")
    console.print()

    # Create agent (reused across turns)
    # Import Rich trace middleware for message display
    from genai_tk.extra.agents.deer_flow.config_thread_data_middleware import ConfigThreadDataMiddleware
    from genai_tk.extra.agents.deer_flow.rich_middleware import DeerFlowRichTraceMiddleware

    with console.status("🦌 Setting up Deer-flow agent...", spinner="dots"):
        checkpointer = MemorySaver()

        # Always create Rich trace middleware - it's needed for clean tool call display
        # In verbose mode, it still shows tool calls cleanly while DEBUG logs show internals
        trace_middleware = DeerFlowRichTraceMiddleware(console=console)

        # Create thread_data middleware for file I/O support
        # This provides workspace paths AND pre-initializes the local sandbox,
        # enabling tools like write_file, read_file, bash without runtime.context
        thread_data_mw = ConfigThreadDataMiddleware()

        agent = create_deer_flow_agent_simple(
            profile=profile_dict,
            llm=llm,
            checkpointer=checkpointer,
            trace_middleware=trace_middleware,
            thread_data_middleware=thread_data_mw,
            interactive_mode=True,  # Interactive chat mode - includes ClarificationMiddleware
        )

    # Display loaded tools and skills
    _display_agent_tools_and_skills(profile_dict, console)

    thread_id = str(uuid.uuid4())

    # Set up prompt history
    history_file = Path(".deerflow.input.history")
    session = PromptSession(history=FileHistory(str(history_file)))

    # Process initial input if provided
    if initial_input:
        await _process_message(
            user_input=initial_input,
            agent=agent,
            thread_id=thread_id,
            stream_enabled=stream_enabled,
            console=console,
            trace_enabled=True,
            thread_data_middleware=thread_data_mw,
        )
        console.print()  # Add spacing

    # Chat loop
    try:
        while True:
            try:
                # Get user input with prompt_toolkit
                with patch_stdout():
                    prompt_style = Style.from_dict({"prompt": "bold green"})
                    user_input = await session.prompt_async(
                        ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                    )

                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                    break
                elif user_input == "/clear":
                    thread_id = str(uuid.uuid4())
                    console.print("[yellow]Conversation cleared[/yellow]")
                    continue
                elif user_input == "/help":
                    console.print(
                        Panel(
                            "/help   – show this help\n"
                            "/info   – display agent configuration\n"
                            "/quit   – exit chat mode\n"
                            "/clear  – clear conversation history\n"
                            "/trace  – open LangSmith trace in browser",
                            title="[bold cyan]Commands[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    continue
                elif user_input == "/info":
                    _display_agent_info(profile_dict, llm, resolved_llm_id)
                    continue
                elif user_input == "/trace":
                    webbrowser.open("https://smith.langchain.com/")
                    continue
                elif user_input.startswith("/"):
                    console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                    continue

                # Process message through agent
                await _process_message(
                    user_input=user_input,
                    agent=agent,
                    thread_id=thread_id,
                    stream_enabled=stream_enabled,
                    console=console,
                    trace_enabled=True,
                    thread_data_middleware=thread_data_mw,
                )
                console.print()  # Add spacing

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Received keyboard interrupt. Use /quit to exit.[/bold yellow]")
                continue
            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Error: {e}[/red]",
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                    )
                )
                logger.exception("Agent execution error")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
