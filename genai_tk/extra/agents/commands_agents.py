"""CLI commands for AI Extra functionality.

This module provides command-line interface commands for:
- Running MCP React agents
- Executing SmolAgents with custom tools
- Processing PDF files with OCR
- Running Fabric patterns

The commands are registered with a Typer CLI application and provide:
- Input/output handling (including stdin)
- Configuration of LLM parameters
- Tool integration
- Batch processing capabilities
"""

import asyncio
import os
import sys
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config

# ============================================================================
# Deep Agent CLI Helper Functions
# ============================================================================


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


class AgentCommands(CliTopCommand):
    description: str = "Commands to create Autonomus Agents"

    def get_description(self) -> tuple[str, str]:
        return "agents", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("react")
        def react(
            input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
            mcp: Annotated[
                list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
            ] = [],
            config: Annotated[
                Optional[str],
                Option(
                    "--config", "-c", help="Configuration name from react_agent.yaml (e.g. 'Weather', 'Web Research')"
                ),
            ] = None,
            pre_prompt: Annotated[
                Optional[str],
                Option("--pre-prompt", "-p", help="Additional context or instructions to send before the user query"),
            ] = None,
            cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
            lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
            lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
            llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
            chat: Annotated[bool, Option("--chat", "-s", help="Start an interactive chat with agent")] = False,
        ) -> None:
            """
            Run a LangChain ReAct agent with tools and MCP Servers.

            Examples:

            # Using MCP servers directly:
            echo "get news from atos.net web site" | uv run cli mcp-agent --mcp playwright --mcp filesystem

            # Using a predefined configuration:
            uv run cli agents react --config "Weather" "What is the wind force in Toulouse?"
            uv run cli agents react --config "Web Research" "Research the latest AI developments"

            Use --chat to start an interactive shell where you can send multiple prompts to the agent.
            """

            from genai_tk.cli.langchain_agent import (
                run_langchain_agent_direct,
                run_langchain_agent_shell,
            )
            from genai_tk.core.llm_factory import LlmFactory
            from genai_tk.extra.agents.langchain_setup import setup_langchain

            # Handle configuration loading using shared loader
            config_tools = []
            config_mcp_servers = []
            config_pre_prompt = None
            final_llm_id = None

            if config:
                from genai_tk.tools.langchain.shared_config_loader import create_langchain_agent_config

                config_path = global_config().get_dir_path("paths.config")

                agent_config = create_langchain_agent_config(
                    config_file=config_path / "agents/langchain.yaml",
                    config_section="langchain_agents",
                    config_name=config,
                    llm=None,  # Let the config loader handle LLM resolution from YAML
                )

                if agent_config is None:
                    print(f"❌ Error: Configuration '{config}' not found in config/basic/agents/langchain.yaml")
                    print()
                    from genai_tk.cli.config_display import display_react_agent_configs

                    display_react_agent_configs()
                    return

                # Extract configuration parameters
                config_tools = agent_config.tools  # Already processed BaseTool instances
                config_mcp_servers = agent_config.mcp_servers
                config_pre_prompt = agent_config.pre_prompt

                if llm:
                    # CLI option takes precedence
                    try:
                        final_llm_id = LlmFactory.resolve_llm_identifier(llm)
                        print(f"Using LLM from CLI: {final_llm_id}")
                    except ValueError as e:
                        print(f"Error: {e}")
                        return
                elif agent_config.llm_id:
                    # Use LLM from configuration
                    final_llm_id = agent_config.llm_id
                    print(f"Using LLM from config: {final_llm_id}")
                # else: final_llm_id remains None, will use default
            else:
                # No config specified, just resolve CLI LLM if provided
                if llm:
                    try:
                        final_llm_id = LlmFactory.resolve_llm_identifier(llm)
                        print(f"Using LLM from CLI: {final_llm_id}")
                    except ValueError as e:
                        print(f"Error: {e}")
                        return

                print(f"Using ReAct configuration '{config}':")
                if config_tools:
                    tool_names = [getattr(t, "name", str(type(t).__name__)) for t in config_tools]
                    print(f"  Tools: {', '.join(tool_names)}")
                if config_mcp_servers:
                    print(f"  MCP servers: {', '.join(config_mcp_servers)}")

            # Merge MCP servers from config and command line
            final_mcp_servers = list(set(mcp + config_mcp_servers))
            # Determine final pre_prompt (command line takes precedence over config)
            final_pre_prompt = pre_prompt or config_pre_prompt

            setup_langchain(final_llm_id, lc_debug, lc_verbose, cache)

            if chat:
                asyncio.run(
                    run_langchain_agent_shell(
                        final_llm_id,
                        tools=config_tools,
                        mcp_server_names=final_mcp_servers,
                        system_prompt=final_pre_prompt,
                    )
                )
            else:
                if not input and not sys.stdin.isatty():
                    input = sys.stdin.read()
                if not input or len(input.strip()) < 1:
                    print("Error: Input parameter or something in stdin is required")
                    return

                asyncio.run(
                    run_langchain_agent_direct(
                        input,
                        llm_id=final_llm_id,
                        mcp_server_names=final_mcp_servers,
                        additional_tools=config_tools,
                        pre_prompt=final_pre_prompt,
                    )
                )

        @cli_app.command("smol")
        def smol(
            input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
            tools: Annotated[
                list[str], Option("--tools", "-t", help="Tools to use (web_search, calculator, etc.)")
            ] = [],
            config: Annotated[
                Optional[str],
                Option("--config", "-c", help="Configuration name from codeact_agent.yaml (e.g. 'Titanic', 'MCP')"),
            ] = None,
            pre_prompt: Annotated[
                Optional[str],
                Option("--pre-prompt", "-p", help="Additional context or instructions to send before the user query"),
            ] = None,
            llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
            imports: list[str] | None = None,
            chat: Annotated[bool, Option("--chat", "-s", help="Start an interactive shell to send prompts")] = False,
        ) -> None:
            """
            Run a SmolAgent CodeAct agent with tools.

            Examples:

            # Using tools directly:
            uv run cli agents smol --input "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search
            echo "Tell me about machine learning" | uv run cli agents smol -t web_search

            # Using a predefined configuration:
            uv run cli agents smol --config "Titanic" --input "What is the proportion of female passengers that survived?"
            uv run cli agents smol --config "MCP" --input "What is the current weather in Toulouse?"

            Use --chat to start an interactive shell where you can send multiple prompts to the agent.
            """
            from smolagents import CodeAgent, Tool
            from smolagents.default_tools import TOOL_MAPPING

            from genai_tk.core.llm_factory import LlmFactory
            from genai_tk.extra.agents.langchain_setup import setup_langchain
            from genai_tk.tools.smolagents.config_loader import (
                CONF_YAML_FILE,
                convert_langchain_tools,
                load_smolagent_demo_config,
                process_tools_from_config,
            )

            # Resolve LLM identifier if provided
            llm_id = None
            if llm:
                try:
                    llm_id = LlmFactory.resolve_llm_identifier(llm)
                except ValueError as e:
                    print(f"Error: {e}")
                    return

            if not setup_langchain(llm_id):
                return

            # Handle configuration loading
            config_tools = []
            config_authorized_imports = []
            config_pre_prompt = None
            final_tools = []
            final_imports = imports or []

            if config:
                demo_config = load_smolagent_demo_config(config)
                if demo_config is None:
                    print(f"❌ Error: Configuration '{config}' not found in {CONF_YAML_FILE}")
                    print()
                    from genai_tk.cli.config_display import display_smolagents_configs

                    display_smolagents_configs()
                    return

                # Extract configuration parameters
                raw_config_tools = process_tools_from_config(demo_config.get("tools", []))
                config_tools = convert_langchain_tools(raw_config_tools)  # Convert LangChain tools to SmolAgent tools
                config_authorized_imports = demo_config.get("authorized_imports", [])
                config_pre_prompt = demo_config.get("pre_prompt")

                print(f"Using CodeAct configuration '{config}':")
                if config_tools:
                    tool_names = [getattr(t, "name", str(type(t).__name__)) for t in config_tools]
                    print(f"  Tools: {', '.join(tool_names)}")
                if config_authorized_imports:
                    print(f"  Authorized imports: {', '.join(config_authorized_imports)}")

                # Use config tools and imports
                final_tools.extend(config_tools)
                final_imports.extend(config_authorized_imports)
            # Determine final pre_prompt (command line takes precedence over config)
            final_pre_prompt = pre_prompt or config_pre_prompt

            model = LlmFactory(llm=llm_id).get_smolagent_model()
            available_tools = final_tools.copy()

            # Add command-line specified tools
            for tool_name in tools:
                if "/" in tool_name:
                    available_tools.append(Tool.from_space(tool_name))
                else:
                    if tool_name in TOOL_MAPPING:
                        available_tools.append(TOOL_MAPPING[tool_name]())
                    else:
                        raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

            tool_display = tools + [getattr(t, "name", str(type(t).__name__)) for t in config_tools]
            print(f"Running agent with these tools: {tool_display}")

            if chat:
                print("Chat mode for smolagents is not yet implemented")
                return
                # asyncio.run(run_smolagent_shell(llm_id, mcp_servers=[]))
            else:
                # Handle input from --input parameter or stdin
                if not input and not sys.stdin.isatty():
                    input = sys.stdin.read()
                if not input or len(input.strip()) < 1:
                    print("Error: Input parameter or something in stdin is required")
                    return

                # Create agent with optional pre_prompt as instructions
                agent_kwargs = {
                    "tools": available_tools,
                    "model": model,
                    "additional_authorized_imports": final_imports,
                }
                if final_pre_prompt:
                    agent_kwargs["instructions"] = final_pre_prompt
                agent = CodeAgent(**agent_kwargs)
                agent.run(input)

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
            import asyncio

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
                console.print(
                    "[red]Error:[/red] Input text required (or use --chat for interactive mode)", style="bold"
                )
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

        @cli_app.command("deerflow")
        def deerflow(
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
            web: Annotated[
                bool,
                typer.Option(
                    "--web",
                    help=(
                        "Write deer-flow config.yaml and extensions_config.json, "
                        "then print the 'make dev' command to start the web interface."
                    ),
                ),
            ] = False,
        ) -> None:
            """Run Deer-flow agents with advanced reasoning capabilities.

            Deer-flow agents offer advanced capabilities:
            - Subagent orchestration (parallel task delegation)
            - Thinking mode (enhanced reasoning)
            - Planning mode (multi-step task planning)
            - Skills system (loadable workflows)

            Note: When using CLI integration, some deer-flow middlewares are automatically
            disabled because they require deer-flow's full runtime infrastructure (which provides
            runtime.context). This affects both single-shot and chat modes:
            - ThreadDataMiddleware, UploadsMiddleware, TitleMiddleware, MemoryMiddleware
            - ClarificationMiddleware (single-shot mode only)

            These limitations mean skills requiring file I/O (ppt-generation, image-generation, etc.)
            may not work properly even in chat mode. For full functionality, use deer-flow natively.

            Examples:
                # List available profiles
                cli agents deerflow --list

                # Run with a profile
                cli agents deerflow -p "Research Assistant" "Explain quantum computing"

                # Interactive chat mode (recommended for skills like ppt-generation)
                cli agents deerflow -p "Research Assistant" --chat

                # With LLM override
                cli agents deerflow -p "Coder" --llm gpt_41_openrouter "Write a sorting algorithm"

                # Stream intermediate steps
                cli agents deerflow -p "Research Assistant" --stream "What are AI trends?"

                # Enable verbose logging for debugging
                cli agents deerflow -p "Research Assistant" --verbose "Complex query"

                # Add extra MCP servers
                cli agents deerflow -p "Coder" --mcp math --mcp weather "Calculate weather patterns"

                # Override mode
                cli agents deerflow -p "Web Browser" --mode ultra "Research topic"

                # Read from stdin
                echo "What is RAG?" | cli agents deerflow -p "Research Assistant"

                # Write config files and print make dev instruction
                cli agents deerflow -p "Research Assistant" --web
            """
            from genai_tk.extra.agents.deer_flow.cli_commands import (
                _get_default_profile_name,
                _list_profiles,
                _run_chat_mode,
                _run_single_shot,
            )

            # Configure logging level based on verbose flag
            if verbose:
                logger.remove()  # Remove default handler
                logger.add(
                    sys.stderr,
                    level="DEBUG",
                    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
                )
                logger.debug("Verbose logging enabled")

            # Handle --list flag
            if list_profiles:
                _list_profiles()
                return

            # Use default profile if not specified
            if not profile:
                default_profile = _get_default_profile_name()
                if default_profile:
                    profile = default_profile
                    print(f"Using default profile: {profile}")
                else:
                    print("❌ Error: --profile/-p is required (or use --list to see available profiles)")
                    raise typer.Exit(1)

            # Handle --web: write config files and instruct user to start the web interface
            if web:
                from rich.console import Console as _Console

                from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
                from genai_tk.extra.agents.deer_flow.profile import (
                    DeerFlowError,
                    load_deer_flow_profiles,
                    validate_mcp_servers,
                    validate_mode,
                    validate_profile_name,
                )
                from genai_tk.utils.config_mngr import global_config

                console = _Console()
                config_dir = global_config().get_dir_path("paths.config")
                config_path_str = str(config_dir / "agents" / "deerflow.yaml")

                try:
                    profiles = load_deer_flow_profiles(config_path_str)
                    df_profile = validate_profile_name(profile, profiles)
                except DeerFlowError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise typer.Exit(1) from e

                try:
                    if mode:
                        df_profile.mode = validate_mode(mode)
                    if mcp:
                        validated = validate_mcp_servers(list(mcp))
                        df_profile.mcp_servers = list(set(df_profile.mcp_servers + validated))
                except DeerFlowError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise typer.Exit(1) from e

                from genai_tk.extra.agents.deer_flow.cli_commands import _resolve_model_name

                raw_llm = llm or df_profile.llm
                selected_llm_id = _resolve_model_name(raw_llm) if raw_llm else None

                with console.status("Writing Deer-flow config files...", spinner="dots"):
                    yaml_path, ext_path = setup_deer_flow_config(
                        mcp_server_names=df_profile.mcp_servers,
                        skill_directories=df_profile.skill_directories,
                        sandbox=df_profile.sandbox,
                        selected_llm=selected_llm_id,
                    )

                console.print(f"[green]✔[/green] Config written:     [cyan]{yaml_path}[/cyan]")
                df_path = df_profile.deer_flow_path or os.environ.get("DEER_FLOW_PATH", "")
                if df_path:
                    import pathlib

                    root_copy = pathlib.Path(df_path) / "config.yaml"
                    console.print(
                        f"[green]✔[/green] Config copied to:   [cyan]{root_copy}[/cyan] [dim](project root — preferred location)[/dim]"
                    )
                console.print(f"[green]✔[/green] Extensions written: [cyan]{ext_path}[/cyan]")
                console.print()
                if df_path:
                    console.print(
                        "[yellow]⚠[/yellow]  If deer-flow is already running, [bold]stop it first[/bold] — "
                        "the backend caches the config at startup and won't pick up changes automatically."
                    )
                    console.print()
                    console.print(
                        f"Then run [bold cyan]make dev[/bold cyan] in [dim]{df_path}[/dim] to start the web interface."
                    )
                else:
                    console.print(
                        "[yellow]⚠[/yellow]  Stop any running deer-flow instance first, then run "
                        "[bold cyan]make dev[/bold cyan] in your deer-flow directory."
                    )
                return

            # Get input from stdin if not provided
            if not input_text and not chat:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read().strip()
                    if not input_text:
                        print("❌ Error: No input provided")
                        raise typer.Exit(1)
                else:
                    print("⚠️  Warning: No input provided. Use positional argument, stdin, or --chat mode.")
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
                    # Type guard: at this point input_text is guaranteed to be str (not None)
                    # because the check above ensures it's either provided or read from stdin
                    if input_text is None:
                        raise ValueError("input_text should be non-None for non-chat mode")
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
                print("\n⚠️  Interrupted by user")
                raise typer.Exit(0) from None
            except typer.Exit:
                # Clean exit with already-displayed error message
                raise
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Deer-flow agent error")
                raise typer.Exit(1) from e
