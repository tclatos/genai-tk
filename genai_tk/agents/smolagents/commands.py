"""SmolAgents CodeAct CLI commands."""

from __future__ import annotations

import sys
from typing import Annotated, Optional

import typer
from typer import Option


def register(cli_app: typer.Typer) -> None:
    """Register the ``smol`` command with *cli_app*."""

    @cli_app.command("smol")
    def smol(
        input_text: Annotated[
            Optional[str],
            typer.Argument(help="Query text. Omit to read from stdin or use --input."),
        ] = None,
        input: Annotated[
            str | None, typer.Option("--input", "-i", help="Input query (alternative to positional arg)")
        ] = None,
        tools: Annotated[list[str], Option("--tools", "-t", help="Tools to use (web_search, calculator, etc.)")] = [],
        config: Annotated[
            Optional[str],
            Option("--config", "-c", help="Configuration name from codeact_agent.yaml (e.g. 'Titanic', 'MCP')"),
        ] = None,
        pre_prompt: Annotated[
            Optional[str],
            Option("--pre-prompt", "-p", help="Additional context or instructions to send before the user query"),
        ] = None,
        llm: Annotated[str, Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = "default",
        imports: list[str] | None = None,
        chat: Annotated[bool, Option("--chat", "-s", help="Start an interactive shell to send prompts")] = False,
        sandbox: Annotated[
            Optional[str],
            Option("--sandbox", "-b", help="Sandbox type: local (default), docker, e2b"),
        ] = None,
    ) -> None:
        """
        Run a SmolAgent CodeAct agent with tools.

        Examples:

        # Using tools directly:
        uv run cli agents smol "How long to run through Pont des Arts at leopard speed?" -t web_search
        uv run cli agents smol --input "..." -t web_search
        echo "Tell me about machine learning" | uv run cli agents smol -t web_search

        # Using a predefined configuration:
        uv run cli agents smol --config "Titanic" "What is the proportion of female passengers that survived?"
        uv run cli agents smol --config "MCP" "What is the current weather in Toulouse?"

        Use --chat to start an interactive shell where you can send multiple prompts to the agent.
        """
        from smolagents import CodeAgent, Tool
        from smolagents.default_tools import TOOL_MAPPING

        from genai_tk.agents.langchain.setup import setup_langchain
        from genai_tk.core.llm_factory import LlmFactory
        from genai_tk.tools.smolagents.config_loader import (
            CONF_YAML_FILE,
            create_demo_from_config,
        )

        # Resolve LLM identifier
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
            demo = create_demo_from_config(config)
            if demo is None:
                print(f"❌ Error: Configuration '{config}' not found in {CONF_YAML_FILE}")
                print()
                from genai_tk.cli.config_display import display_smolagents_configs

                display_smolagents_configs()
                return

            # Extract configuration parameters
            config_tools = demo.tools
            config_authorized_imports = demo.authorized_imports
            config_pre_prompt = demo.pre_prompt

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
        else:
            # Positional arg takes priority; --input/-i is the fallback
            query = input_text or input
            # Handle input from stdin
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read()
            if not query or len(query.strip()) < 1:
                print("Error: Provide a query as a positional argument, --input/-i, or via stdin")
                return

            # Create agent with optional pre_prompt as instructions
            agent_kwargs: dict = {
                "tools": available_tools,
                "model": model,
                "additional_authorized_imports": final_imports,
            }
            if final_pre_prompt:
                agent_kwargs["instructions"] = final_pre_prompt

            if sandbox and sandbox != "local":
                if sandbox == "docker":
                    from genai_tk.agents.sandbox.config import get_docker_smol_settings

                    smol_cfg = get_docker_smol_settings()
                    agent_kwargs["executor_type"] = "docker"
                    agent_kwargs["executor_kwargs"] = {
                        "image_name": smol_cfg.image,
                        "container_run_kwargs": {
                            "mem_limit": smol_cfg.mem_limit,
                            "cpu_quota": smol_cfg.cpu_quota,
                            "pids_limit": smol_cfg.pids_limit,
                        },
                    }
                    print(f"Using Docker sandbox (image: {smol_cfg.image})")
                elif sandbox == "e2b":
                    from genai_tk.agents.sandbox.config import get_e2b_settings

                    e2b_cfg = get_e2b_settings()
                    if not e2b_cfg.api_key:
                        print("Error: E2B_API_KEY is not set. Export it or configure it in sandbox.yaml.")
                        return
                    agent_kwargs["executor_type"] = "e2b"
                    print("Using E2B sandbox")
                elif sandbox == "wasm":
                    print("Error: WASM sandbox is not yet supported.")
                    return
                else:
                    print(f"Error: Unknown sandbox type '{sandbox}'. Valid values: local, docker, e2b")
                    return

            agent = CodeAgent(**agent_kwargs)
            agent.run(query)
