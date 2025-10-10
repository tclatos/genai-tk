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
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from typer import Option

from genai_tk.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def config_info() -> None:
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

        console.print(Panel(f"[bold blue]Selected configuration:[/bold blue] {config.selected_config}", expand=False))

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
                        test_store = kv_registry.get(store_id)
                        status = "[green]✓ available[/green]"
                    except Exception as e:
                        error_msg = str(e)
                        if len(error_msg) > 20:
                            error_msg = f"{error_msg[:17]}..."
                        status = f"[red]✗ error[/red]"
                        
                    kv_stores_table.add_row(store_id, store_type, config_display, status)
                    
                except Exception as e:
                    # Handle individual store errors
                    kv_stores_table.add_row(store_id, "Error", str(e)[:25], "[red]✗ error[/red]")
            
            if not available_stores:
                kv_stores_table.add_row(
                    "[dim]No stores configured[/dim]",
                    "[dim]N/A[/dim]", 
                    "[dim]Configure in config file[/dim]",
                    "[dim]N/A[/dim]"
                )
                
            console.print(kv_stores_table)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load KV store information: {e}[/yellow]")

        console.print(f"LangSmith Tracing: {ls_utils.tracing_is_enabled()}")
        # # Deep Agents info

        # agents_table = Table(title="Deep Agents", show_header=True, header_style="bold magenta")
        # agents_table.add_column("Type", style="cyan")
        # agents_table.add_column("Description", style="green")
        # agents_table.add_column("Features", style="yellow")

        # agents_table.add_row("Research", "Comprehensive research with web search", "Planning, Search, Notes, Reports")
        # agents_table.add_row("Coding", "Write, debug, and refactor code", "Sub-agents, Testing, Documentation")
        # agents_table.add_row("Analysis", "Data analysis and insights", "Exploration, Patterns, Reports")
        # agents_table.add_row("Custom", "User-defined agent with custom instructions", "Fully configurable")

        # console.print(agents_table)

    @cli_app.command()
    def llm(
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        temperature: Annotated[
            float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
        ] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        reasoning: Annotated[
            bool, Option("--reasoning", help="Enable reasoning/thinking mode (for compatible models)")
        ] = False,
        raw: Annotated[bool, Option("--raw", "-r", help="Output raw LLM response object")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
    ) -> None:
        """
        Invoke an LLM.

        input can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm. This can be either an LLM ID or a tag defined in config (e.g., 'fake', 'powerful_model').
        If not specified, the default model is used.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.

        Examples:
            uv run cli llm "Tell me a joke" --llm fake
            uv run cli llm "Explain AI" --llm parrot_local_fake
        """

        from langchain_core.output_parsers import StrOutputParser
        from rich import print as pprint

        from genai_tk.core.llm_factory import LlmFactory
        from genai_tk.extra.agents.langchain_setup import setup_langchain

        # For compatibility with setup_langchain, resolve the llm to an llm_id if provided
        llm_id = None
        if llm:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        # Check if executed as part ot a pipe
        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        llm_factory = LlmFactory.from_unified_parameter(
            llm=llm,
            json_mode=False,
            streaming=stream,
            reasoning=reasoning,
            cache=cache,
            llm_params={"temperature": temperature},
        )
        llm_model = llm_factory.get()
        if raw:
            if stream:
                for chunk in llm_model.stream(input):
                    pprint(chunk)
            else:
                result = llm_model.invoke(input)
                pprint(result)
        else:
            chain = llm_model | StrOutputParser()
            if stream:
                for s in chain.stream(input):
                    print(s, end="", flush=True)
                print("\n")
            else:
                result = chain.invoke(input)
                pprint(result)

    @cli_app.command()
    def run(
        runnable_name: Annotated[str, typer.Argument(help="Name of registered Runnable to execute")],
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        path: Annotated[Path | None, typer.Option(help="File path input for the chain")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        temperature: Annotated[
            float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
        ] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        reasoning: Annotated[
            bool, Option("--reasoning", help="Enable reasoning/thinking mode (for compatible models)")
        ] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
    ) -> None:
        """
        Run a Runnable or directly invoke an LLM.

        If no runnable_name is provided, uses the default LLM to directly process the input, that
        can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm. This can be either an LLM ID or a tag defined in config (e.g., 'fake', 'powerful_model').
        If not specified, the default model is used.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.

        Examples:
            uv run cli run joke --input "bears"
            uv run cli run joke --input "bears" --llm fake
            uv run cli run joke --input "bears" --llm parrot_local_fake
        """

        from devtools import pprint

        from genai_tk.core.chain_registry import ChainRegistry
        from genai_tk.core.llm_factory import LlmFactory
        from genai_tk.extra.agents.langchain_setup import setup_langchain
        from genai_tk.utils.config_mngr import global_config

        # For compatibility with setup_langchain, resolve the llm to an llm_id if provided
        llm_id = None
        if llm:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        # Handle input from stdin if no input parameter provided
        if not input and not sys.stdin.isatty():  # Check if stdin has data (pipe/redirect)
            input = str(sys.stdin.read())
            if len(input.strip()) < 3:  # Ignore very short inputs
                input = None

        chain_registry = ChainRegistry.instance()
        ChainRegistry.load_modules()
        # If runnable_name is provided, proceed with existing logic
        runnables_list = sorted([f"'{o.name}'" for o in chain_registry.get_runnable_list()])
        runnables_list_str = ", ".join(runnables_list)
        runnable_item = chain_registry.find(runnable_name)
        if runnable_item:
            first_example = runnable_item.examples[0]
            llm_args = {"temperature": temperature}
            if reasoning:
                llm_args["reasoning"] = reasoning
            # Use the resolved llm_id or default
            try:
                final_llm_id = llm_id or global_config().get_str("llm.models.default")
            except Exception as e:
                print(f"Error: {e}")
                return

            config = {
                "llm": final_llm_id,
                "llm_args": llm_args,
            }
            if path:
                config |= {"path": path}
            elif first_example.path:
                config |= {"path": first_example.path}
            if not input:
                input = first_example.query[0]

            chain = runnable_item.get().with_config(configurable=config)
        else:
            print(f"Runnable '{runnable_name}' not found in config. Should be in: {runnables_list_str}")
            return

        if stream:
            for s in chain.stream(input):
                print(s, end="", flush=True)
            print("\n")
        else:
            result = chain.invoke(input)
            pprint(result)

    @cli_app.command()
    def list_models() -> None:
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

    @cli_app.command()
    def embedd(
        input: Annotated[str, typer.Argument(help="Text to embed")],
        model_id: Annotated[Optional[str], Option("--model-id", "-m", help="Embeddings model ID")] = None,
    ) -> None:
        """
        Invoke an embedder.

        ex: uv run cli embedd "string to be embedded"
        """

        from rich.console import Console
        from rich.table import Table

        from genai_tk.core.embeddings_factory import EmbeddingsFactory

        if model_id is not None:
            if model_id not in EmbeddingsFactory.known_items():
                print(f"Error: {model_id} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                return
            global_config().set("llm.models.default", model_id)

        factory = EmbeddingsFactory(
            embeddings_id=model_id,
        )
        embedder = factory.get()
        vector = embedder.embed_documents([input])

        console = Console()
        table = Table(title="Embeddings Summary", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", factory.embeddings_id or "default")
        table.add_row("Vector Length", str(len(vector[0])))
        table.add_row("First 40 Elements", ", ".join(f"{x:.4f}" for x in vector[0][:40]) + " [...]")

        console.print(table)

    @cli_app.command()
    def list_mcp_tools(
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

    @cli_app.command()
    def similarity(
        sentences: Annotated[list[str], typer.Argument(help="List of sentences to compare (first is reference)")],
        model_id: Annotated[Optional[str], Option("--model-id", "-m", help="Embeddings model ID")] = None,
    ) -> None:
        """
        Calculate semantic similarity between sentences using cosine similarity.

        The first sentence is used as reference and compared to the others.

        Example:
            uv run cli similarity "This is a test" "This is another test" "Completely different"
        """
        from langchain_community.utils.math import cosine_similarity

        from genai_tk.core.embeddings_factory import EmbeddingsFactory, get_embeddings

        if len(sentences) < 2:
            print("Error: At least 2 sentences are required")
            return

        if model_id is not None:
            if model_id not in EmbeddingsFactory.known_items():
                print(f"Error: {model_id} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                return

        embedder = get_embeddings(embeddings_id=model_id)

        # Generate embeddings for all sentences
        vectors = embedder.embed_documents(sentences)

        # Calculate similarity between first sentence and others
        reference_vector = [vectors[0]]
        other_vectors = vectors[1:]

        similarities = cosine_similarity(reference_vector, other_vectors)

        # Display results in table format
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Semantic Similarity", show_header=True, header_style="bold blue")
        table.add_column("Reference Sentence", style="cyan")
        table.add_column("Comparison Sentence", style="green")
        table.add_column("Score", style="magenta", justify="right")

        for i, score in enumerate(similarities[0]):
            table.add_row(sentences[0], sentences[i + 1], f"{score:.3f}")

        console.print(table)

    @cli_app.command()
    def list_mcp_prompts(
        filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter prompts by server names")] = None,
    ) -> None:
        """Display information about available MCP prompts.

        Shows the list of prompts from MCP servers along with their descriptions.
        Can be filtered by server names.

        Example:
            uv run cli list-mcp-prompts
            uv run cli list-mcp-prompts --filter server1 server2
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
