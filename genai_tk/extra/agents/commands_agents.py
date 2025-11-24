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
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config


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
                if not input or len(input) < 5:
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
                if not input or len(input) < 5:
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
            input: Annotated[str | None, typer.Argument(help="Input query or '-' to read from stdin")] = None,
            config: Annotated[
                Optional[str],
                Option("--config", "-c", help="Configuration name from deep_agent.yaml (e.g. 'Research', 'Coding')"),
            ] = None,
            stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
            llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag)")] = None,
            instructions: Annotated[
                Optional[str], Option("--instructions", "-i", help="Custom instructions for the agent")
            ] = None,
            tools: Annotated[Optional[list[str]], Option("--tools", "-t", help="Additional tools to include")] = None,
            files: Annotated[
                Optional[list[Path]], Option("--files", "-f", help="Files to include in agent context")
            ] = None,
            output_dir: Annotated[
                Optional[Path], Option("--output-dir", "-o", help="Directory to save agent outputs")
            ] = None,
        ) -> None:
            """
            Run a Deep Agent for complex AI tasks.

            Deep Agents combine planning, file system access, and sub-agents for comprehensive task execution.

            Examples:

            # Using a predefined configuration:
            uv run cli deep-agent --config "Research" "Latest developments in quantum computing"
            uv run cli deep-agent --config "Coding" "Write a Python function to calculate Fibonacci"
            uv run cli deep-agent --config "Data Analysis" --files data.csv "Analyze this dataset"

            # Using custom instructions:
            uv run cli deep-agent --instructions "You are a helpful assistant" "Help me with this task"

            # Reading from stdin:
            echo "Debug this code" | uv run cli deep-agent --config "Coding"
            """
            import asyncio

            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn

            from genai_tk.core.deep_agents import DeepAgentConfig, deep_agent_factory, run_deep_agent
            from genai_tk.core.llm_factory import LlmFactory
            from genai_tk.tools.smolagents.deep_config_loader import (
                DEEP_AGENT_CONF_YAML_FILE,
                load_deep_agent_demo_config,
            )

            # Get input from stdin if needed
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()
            if not input or len(input) < 5:
                print("Error: Input parameter or something in stdin is required")
                return

            console = Console()

            # Handle configuration loading
            demo_config = None
            config_instructions = None
            config_tools = []
            config_mcp_servers = []
            config_enable_file_system = True
            config_enable_planning = True

            if config:
                demo_config = load_deep_agent_demo_config(config)
                if demo_config is None:
                    print(f"Error: Configuration '{config}' not found in {DEEP_AGENT_CONF_YAML_FILE}")
                    return

                # Extract configuration parameters
                config_instructions = demo_config.get("instructions", "")
                config_tools = demo_config.get("tools", [])
                config_mcp_servers = demo_config.get("mcp_servers", [])
                config_enable_file_system = demo_config.get("enable_file_system", True)
                config_enable_planning = demo_config.get("enable_planning", True)

                print(f"Using Deep Agent configuration '{config}':")
                if config_tools:
                    print(f"  Tools: {len(config_tools)} tool(s) configured")
                if config_mcp_servers:
                    print(f"  MCP servers: {', '.join(config_mcp_servers)}")

            # Use config instructions if provided, otherwise use command line instructions
            final_instructions = instructions or config_instructions

            if not final_instructions:
                if not config:
                    print("Error: Either --config or --instructions must be provided")
                    return
                else:
                    print(f"Error: Configuration '{config}' has no instructions defined")
                    return

            # Load files if provided
            file_contents = {}
            if files:
                for file_path in files:
                    if file_path.exists():
                        file_contents[file_path.name] = file_path.read_text()
                    else:
                        console.print(f"[yellow]Warning: File {file_path} not found[/yellow]")

            # Resolve LLM identifier if provided
            llm_id = None
            if llm:
                try:
                    llm_id = LlmFactory.resolve_llm_identifier(llm)
                except ValueError as e:
                    print(f"Error: {e}")
                    return

            async def run_agent():
                # Set the model FIRST if specified, before creating any agents
                if llm_id:
                    console.print(f"[cyan]Using model: {llm_id}[/cyan]")
                    deep_agent_factory.set_default_model(llm_id)

                # Create agent configuration
                agent_config = DeepAgentConfig(
                    name=f"CLI {config or 'Custom'} Agent",
                    instructions=final_instructions,
                    enable_file_system=config_enable_file_system,
                    enable_planning=config_enable_planning,
                    model=llm_id,
                )

                # Process tools from configuration and convert SmolAgent tools to LangChain tools
                agent_tools = []
                if config_tools:
                    try:
                        from langchain_core.tools import tool
                        from smolagents import Tool as SmolAgentTool

                        from genai_tk.tools.smolagents.config_loader import process_tools_from_config

                        # Process tools using the smolagents processor
                        processed_tools = process_tools_from_config(config_tools)

                        # Convert SmolAgent tools to LangChain tools
                        for tool_instance in processed_tools:
                            if isinstance(tool_instance, SmolAgentTool):
                                # Convert SmolAgent tool to LangChain tool
                                try:
                                    # Create a wrapper function for the SmolAgent tool
                                    def create_langchain_wrapper(smol_tool):
                                        @tool
                                        def langchain_tool_wrapper(query: str) -> str:
                                            """Tool converted from SmolAgent."""
                                            return smol_tool(query)

                                        # Set proper metadata
                                        langchain_tool_wrapper.name = getattr(
                                            smol_tool, "name", type(smol_tool).__name__
                                        )
                                        langchain_tool_wrapper.description = getattr(
                                            smol_tool, "description", f"Tool: {langchain_tool_wrapper.name}"
                                        )

                                        return langchain_tool_wrapper

                                    langchain_tool = create_langchain_wrapper(tool_instance)
                                    agent_tools.append(langchain_tool)
                                except Exception as convert_ex:
                                    console.print(
                                        f"[yellow]Warning: Failed to convert SmolAgent tool {tool_instance}: {convert_ex}[/yellow]"
                                    )
                            else:
                                # Already a LangChain tool or compatible
                                agent_tools.append(tool_instance)

                        tool_names = [getattr(t, "name", str(type(t).__name__)) for t in agent_tools]
                        console.print(f"[cyan]Loaded {len(agent_tools)} tools: {', '.join(tool_names)}[/cyan]")
                    except Exception as ex:
                        console.print(f"[yellow]Warning: Failed to process some tools from config: {ex}[/yellow]")

                # Create the agent
                agent = deep_agent_factory.create_agent(config=agent_config, tools=agent_tools, async_mode=True)

                # Run the agent
                messages = [{"role": "user", "content": input}]

                # Show the user's query in a nice format
                console.print("\n[bold magenta]👤 User Query:[/bold magenta]")
                console.print(f"[italic white]{input}[/italic white]\n")

                with Progress(
                    SpinnerColumn(spinner_name="dots", style="cyan"),
                    TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                    console=console,
                ) as progress:
                    agent_name = config or "Custom"
                    agent_emoji = {
                        "research": "🔍",
                        "coding": "💻",
                        "data analysis": "📊",
                        "web research": "🌐",
                        "documentation writer": "📝",
                        "stock analysis": "📈",
                    }.get(agent_name.lower(), "🤖")

                    task = progress.add_task(f"{agent_emoji} {agent_name} agent is thinking...", total=None)

                    try:
                        result = await run_deep_agent(
                            agent=agent,
                            messages=messages,
                            files=file_contents if file_contents else None,
                            stream=stream,
                        )

                        progress.update(task, description=f"{agent_emoji} {agent_name} agent completed!")
                        progress.update(task, completed=True)
                    except Exception:
                        progress.update(task, description=f"❌ Error in {agent_name} agent")
                        progress.update(task, completed=True)
                        raise

                # Display results with enhanced markdown rendering
                if "messages" in result and result["messages"]:
                    # Get the response content
                    response_content = result["messages"][-1].content

                    # Use Rich's Markdown rendering for better display
                    from rich.markdown import Markdown

                    try:
                        # Render as markdown for better formatting
                        md = Markdown(response_content)
                        # Optionally wrap in a panel for better visual separation
                        # console.print(Panel(md, border_style="cyan", padding=(1, 2)))
                        console.print(md)
                    except Exception as e:
                        # Fallback to plain text if markdown parsing fails
                        logger.warning(f"Markdown rendering failed: {e}")
                        console.print(response_content)

                # Save files if output directory specified
                if output_dir and "files" in result:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    console.print("[bold yellow]📁 Saving files...[/bold yellow]")
                    for filename, content in result["files"].items():
                        file_path = output_dir / filename
                        file_path.write_text(content)
                        console.print(f"  [green]✓[/green] Saved: [cyan]{file_path}[/cyan]")

            # Run the async function
            asyncio.run(run_agent())

        # @cli_app.command("list-deep")
        def list_deep_old() -> None:
            """
            List available Deep Agent configurations and created agents.
            """
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            from genai_tk.core.deep_agents import deep_agent_factory
            from genai_tk.tools.smolagents.deep_config_loader import load_all_deep_agent_demos_from_config

            console = Console()

            # Show available configurations
            configs = load_all_deep_agent_demos_from_config()
            if configs:
                config_table = Table(
                    title="Available Deep Agent Configurations", show_header=True, header_style="bold magenta"
                )
                config_table.add_column("Configuration Name", style="cyan")
                config_table.add_column("Tools", style="green")
                config_table.add_column("MCP Servers", style="yellow")
                config_table.add_column("Examples", style="blue", max_width=40)

                for config in configs:
                    # Show tool count if tools are configured
                    tools_info = f"{len(config.tools)} tool(s)" if config.tools else "No tools"

                    # Show MCP servers
                    mcp_info = ", ".join(config.mcp_servers) if config.mcp_servers else "None"

                    # Show first example or "None"
                    examples_info = config.examples[0] if config.examples else "None"
                    if len(examples_info) > 37:  # Account for "..."
                        examples_info = examples_info[:37] + "..."

                    config_table.add_row(config.name, tools_info, mcp_info, examples_info)

                console.print(config_table)
                console.print()

            # Show created agents
            agents = deep_agent_factory.list_agents()
            if agents:
                agent_table = Table(title="Created Deep Agents", show_header=True, header_style="bold magenta")
                agent_table.add_column("Agent Name", style="cyan")
                agent_table.add_column("Status", style="green")

                for agent_name in agents:
                    agent_table.add_row(agent_name, "Active")

                console.print(agent_table)
            else:
                console.print(
                    Panel(
                        "[yellow]No Deep Agents have been created yet.[/yellow]\n\n"
                        "Use [cyan]uv run cli agents deep --config <config_name> <query>[/cyan] to create and run an agent.",
                        title="Created Deep Agents",
                        border_style="yellow",
                    )
                )
