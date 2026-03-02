"""LangChain ReAct agent CLI commands."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated, Optional

import typer
from typer import Option

from genai_tk.utils.config_mngr import global_config


def register(cli_app: typer.Typer) -> None:
    """Register the ``react`` command with *cli_app*."""

    @cli_app.command("react")
    def react(
        input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
        mcp: Annotated[
            list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
        ] = [],
        config: Annotated[
            Optional[str],
            Option("--config", "-c", help="Configuration name from react_agent.yaml (e.g. 'Weather', 'Web Research')"),
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

        from genai_tk.agents.langchain.agent import (
            run_langchain_agent_direct,
            run_langchain_agent_shell,
        )
        from genai_tk.agents.langchain.setup import setup_langchain
        from genai_tk.core.llm_factory import LlmFactory

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
