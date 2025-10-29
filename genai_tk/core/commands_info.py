"""CLI commands for interacting with AI Core functionality.

This module provides command-line interface commands for:
- Running LLMs directly
- Executing registered Runnable chains
- Getting information about available models and chains
- Working with embeddings

The commands are registered with a Typer CLI application and provide:
- Input/output handling (including stdin)
- Configuration of LLM parameters
- Streaming support
- Caching options
"""

import os
from typing import Annotated

import typer
from typer import Option

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config


class InfoCommands(CliTopCommand):
    """Information and listing commands."""

    description: str = "Information and listing commands."

    def get_description(self) -> tuple[str, str]:
        return "info", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("config")
        def config() -> None:
            """
            Display current configuration, available LLM tags, and API keys.
            """

            from langsmith import utils as ls_utils
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            from genai_tk.core.embeddings_factory import EmbeddingsFactory
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.core.llm_factory import PROVIDER_INFO, LlmFactory

            config = global_config()
            console = Console()

            console.print(
                Panel(f"[bold blue]Selected configuration:[/bold blue] {config.selected_config}", expand=False)
            )

            # Default models info
            default_llm = LlmFactory(llm_id=None)
            default_embeddings = EmbeddingsFactory(embeddings_id=None)
            default_vector_store = EmbeddingsStore.create_from_config("default")

            models_table = Table(title="Default Components", show_header=True, header_style="bold magenta")
            models_table.add_column("Type", style="cyan")
            models_table.add_column("Model ID", style="green")

            models_table.add_row("LLM", str(default_llm.llm_id))
            models_table.add_row("Embeddings", str(default_embeddings.embeddings_id))
            models_table.add_row("Vector-store", str(default_vector_store.backend))

            console.print(models_table)

            # LLM Tags info with enhanced details
            tags_table = Table(
                title="🏷️  LLM Tags (Use these with --llm option)", show_header=True, header_style="bold magenta"
            )
            tags_table.add_column("Tag", style="cyan", width=15)
            tags_table.add_column("LLM ID", style="green", width=25)
            tags_table.add_column("Provider", style="blue", width=12)
            tags_table.add_column("Status", style="yellow", width=12)
            tags_table.add_column("Usage Example", style="dim white", width=30)

            # Get all LLM tags from config under llm.models.*
            llm_models_config = config.get("llm.models", {})
            tag_count = 0
            # Handle both regular dict and OmegaConf DictConfig
            if llm_models_config and hasattr(llm_models_config, "items"):
                for tag, llm_id in llm_models_config.items():
                    if tag != "default":  # Skip the default entry as it's shown above
                        # Check if the LLM ID is available (has API keys and module)
                        is_available = llm_id in LlmFactory.known_items()
                        status = "[green]✓ available[/green]" if is_available else "[red]✗ unavailable[/red]"

                        # Extract provider from LLM ID (last part after underscore)
                        provider = "unknown"
                        if isinstance(llm_id, str) and "_" in llm_id:
                            provider = llm_id.rsplit("_", 1)[-1]

                        # Create usage example
                        example = f"--llm {tag}"

                        tags_table.add_row(tag, str(llm_id), provider, status, example)
                        tag_count += 1

            if tag_count == 0:
                tags_table.add_row(
                    "[dim]No tags configured[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]Configure in config file[/dim]",
                )

            console.print(tags_table)

            # Add helpful usage information
            if tag_count > 0:
                console.print(
                    Panel(
                        "[bold cyan]💡 Usage Tips:[/bold cyan]\n"
                        "• Use tags with [bold]--llm[/bold] option: [green]uv run cli llm 'Hello' --llm fast_model[/green]\n"
                        "• Tags are easier to remember than full LLM IDs\n"
                        "• Configure more tags in your configuration file under [bold]llm.models[/bold]",
                        title="How to use LLM Tags",
                        border_style="cyan",
                        expand=False,
                    )
                )

            # API keys info
            keys_table = Table(title="Available API Keys", show_header=True, header_style="bold magenta")
            keys_table.add_column("Provider", style="cyan")
            keys_table.add_column("Environment Variable", style="green")
            keys_table.add_column("Status", style="yellow")

            for provider, (_, key_name) in PROVIDER_INFO.items():
                if key_name:
                    status = "[green]✓ set[/green]" if key_name in os.environ else "[red]✗ not set[/red]"
                    keys_table.add_row(provider, key_name, status)

            console.print(keys_table)

            # KV Store info
            from genai_tk.extra.kv_store_registry import KvStoreRegistry

            kv_registry = KvStoreRegistry()
            try:
                available_stores = kv_registry.get_available_stores()

                kv_stores_table = Table(title="🗄️  Available KV Stores", show_header=True, header_style="bold magenta")
                kv_stores_table.add_column("Store ID", style="cyan", width=15)
                kv_stores_table.add_column("Type", style="green", width=20)
                kv_stores_table.add_column("Configuration", style="blue", width=30)
                kv_stores_table.add_column("Status", style="yellow", width=15)

                for store_id in available_stores:
                    try:
                        # Try to get configuration details for each store
                        config_info = config.get(f"kv_store.{store_id}", {})

                        # Handle different configuration formats
                        if hasattr(config_info, "get") and "type" in config_info:
                            # New format with explicit type
                            store_type = str(config_info["type"])
                            path_info = config_info.get("path", "N/A")
                            # Truncate long paths for display
                            if isinstance(path_info, str) and len(path_info) > 25:
                                path_info = f"...{path_info[-22:]}"
                            config_display = f"path: {path_info}"
                        elif hasattr(config_info, "get") and "path" in config_info:
                            # Legacy format - infer type
                            path_info = str(config_info["path"])
                            if store_id == "sql" or ("postgresql://" in path_info or "sqlite://" in path_info):
                                store_type = "SQLStore"
                            else:
                                store_type = "LocalFileStore"
                            # Truncate long paths for display
                            if len(path_info) > 25:
                                path_info = f"...{path_info[-22:]}"
                            config_display = f"path: {path_info}"
                        else:
                            # Handle special cases like OmegaConf objects
                            config_str = str(config_info)
                            if "LocalFileStore" in config_str:
                                store_type = "LocalFileStore"
                            elif "SQLStore" in config_str:
                                store_type = "SQLStore"
                            else:
                                store_type = "Unknown"

                            # Try to extract path from string representation
                            if "path" in config_str:
                                import re

                                path_match = re.search(r"'path':\s*'([^']+)'", config_str)
                                if path_match:
                                    path_info = path_match.group(1)
                                    if len(path_info) > 25:
                                        path_info = f"...{path_info[-22:]}"
                                    config_display = f"path: {path_info}"
                                else:
                                    config_display = config_str[:30] + ("..." if len(config_str) > 30 else "")
                            else:
                                config_display = config_str[:30] + ("..." if len(config_str) > 30 else "")

                        # Test if store can be created (indicates proper configuration)
                        try:
                            kv_registry.get(store_id)
                            status = "[green]✓ available[/green]"
                        except Exception as e:
                            error_msg = str(e)
                            if len(error_msg) > 20:
                                error_msg = f"{error_msg[:17]}..."
                            status = "[red]✗ error[/red]"

                        kv_stores_table.add_row(store_id, store_type, config_display, status)

                    except Exception as e:
                        # Handle individual store errors
                        kv_stores_table.add_row(store_id, "Error", str(e)[:25], "[red]✗ error[/red]")

                if not available_stores:
                    kv_stores_table.add_row(
                        "[dim]No stores configured[/dim]",
                        "[dim]N/A[/dim]",
                        "[dim]Configure in config file[/dim]",
                        "[dim]N/A[/dim]",
                    )

                console.print(kv_stores_table)

            except Exception as e:
                console.print(f"[yellow]Warning: Could not load KV store information: {e}[/yellow]")

            console.print(f"LangSmith Tracing: {ls_utils.tracing_is_enabled()}")

        @cli_app.command("models")
        def models() -> None:
            """
            List the known LLMs, embeddings models, and vector stores.
            """
            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel

            from genai_tk.core.embeddings_factory import EmbeddingsFactory
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.core.llm_factory import LlmFactory

            console = Console()

            # Get all items for each category
            llm_items = LlmFactory.known_items()
            embeddings_items = EmbeddingsFactory.known_items()
            vector_items = EmbeddingsStore.known_items()

            # Format LLM items in several columns
            llm_content = Columns([f"• {item}" for item in llm_items], equal=True, expand=True)
            embeddings_content = Columns([f"• {item}" for item in embeddings_items], equal=True, expand=True)
            vector_content = Columns([f"• {item}" for item in vector_items], equal=True, expand=True)
            llm_panel = Panel(llm_content, title="[bold blue]LLMs[/bold blue]", border_style="blue")
            embeddings_panel = Panel(
                embeddings_content,
                title="[bold green]Embeddings[/bold green]",
                border_style="green",
            )
            vector_panel = Panel(
                vector_content,
                title="[bold magenta]Vector Stores[/bold magenta]",
                border_style="magenta",
            )
            console.print(Panel("Available Models & Components", border_style="bright_blue"))
            console.print(llm_panel)
            console.print()
            bottom_row = Columns([embeddings_panel, vector_panel], equal=True, expand=True)
            console.print(bottom_row)

        @cli_app.command("mcp-tools")
        def mcp_tools(
            filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter tools by server names")] = None,
        ) -> None:
            """Display information about available MCP tools.

            Shows the list of tools from MCP servers along with their descriptions.
            Can be filtered by server names.
            """
            import asyncio

            from rich.console import Console
            from rich.table import Table

            from genai_tk.core.mcp_client import get_mcp_tools_info

            async def display_tools():
                tools_info = await get_mcp_tools_info(filter)
                if not tools_info:
                    print("No MCP tools found.")
                    return

                console = Console()
                for server_name, tools in tools_info.items():
                    table = Table(
                        title=f"Server: {server_name}",
                        show_header=True,
                        header_style="bold magenta",
                        row_styles=["", "dim"],
                    )
                    table.add_column("Tool", style="cyan", no_wrap=True)
                    table.add_column("Description", style="green")

                    for tool_name, description in tools.items():
                        table.add_row(tool_name, description)

                    console.print(table)
                    print()  # Add space between tables

            asyncio.run(display_tools())

        @cli_app.command("commands")
        def commands() -> None:
            """Display all registered CLI commands and subcommands.

            Shows a hierarchical view of all available commands in the CLI,
            including their help text. Similar to --help but provides a complete overview.

            Example:
                uv run cli info commands
            """
            from genai_tk.utils.cli.command_tree import display_command_tree

            display_command_tree(
                cli_app,
                title="[bold cyan]📋 CLI Command Structure[/bold cyan]",
            )

        @cli_app.command("mcp-prompts")
        def mcp_prompts(
            filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter prompts by server names")] = None,
        ) -> None:
            """Display information about available MCP prompts.

            Shows the list of prompts from MCP servers along with their descriptions.
            Can be filtered by server names.

            Example:
                uv run cli info mcp-prompts
                uv run cli info mcp-prompts --filter server1 server2
            """
            import asyncio

            from genai_tk.core.mcp_client import get_mcp_prompts

            async def display_prompts():
                prompt_info = await get_mcp_prompts(filter)
                if not prompt_info:
                    print("No MCP tools found.")
                    return

                for server_name, prompts in prompt_info.items():
                    print(f"\nServer: {server_name}")
                    print("-" * (len(server_name) + 8))
                    for name, description in prompts.items():
                        print(f"  {name}:")
                        print(f"    {description}")
                    print()

            asyncio.run(display_prompts())
