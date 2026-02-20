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

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from typer import Option

from genai_tk.core.cache import CacheMethod
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config


class CoreCommands(CliTopCommand):
    """Core commands for interacting with AI models."""

    description: str = "Core commands interacting with AI models."

    def get_description(self) -> tuple[str, str]:
        return "core", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def llm(
            input: Annotated[
                str | None, typer.Option("--input", "-i", help="Input text or '-' to read from stdin")
            ] = None,
            cache: Annotated[CacheMethod, typer.Option(help="Cache strategy")] = "memory",
            temperature: Annotated[
                float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
            ] = 0.0,
            stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
            reasoning: Annotated[
                bool, Option("--reasoning", help="Enable reasoning/thinking mode (for compatible models)")
            ] = False,
            raw: Annotated[bool, Option(help="Output raw LLM response object")] = False,
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
                uv run cli core llm "Tell me a joke" --llm fake
                uv run cli core llm "Explain AI" --llm parrot_local_fake
            """

            from langchain_core.output_parsers import StrOutputParser
            from rich import print as pprint

            from genai_tk.core.llm_factory import LlmFactory
            from genai_tk.extra.agents.langchain_setup import setup_langchain

            # Resolve the llm to get llm_id for setup_langchain
            llm_id = None
            if llm:
                try:
                    llm_id = LlmFactory.resolve_llm_identifier(llm)
                except ValueError as e:
                    print(f"Error: {e}")
                    return

            if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
                return

            # Check if executed as part ot a pipe
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()
            if not input or len(input) < 2:
                print("Error: Input parameter or something in stdin is required")
                return

            llm_factory = LlmFactory(
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
            input: Annotated[
                str | None, typer.Option("--input", "-i", help="Input text or '-' to read from stdin")
            ] = None,
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
                uv run cli core run joke --input "bears"
                uv run cli core run joke --input "bears" --llm fake
                uv run cli core run joke --input "bears" --llm parrot_local_fake
            """

            from devtools import pprint

            from genai_tk.core.chain_registry import ChainRegistry
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
        def embedd(
            input: Annotated[str, typer.Argument(help="Text to embed")],
            model: Annotated[Optional[str], Option("--model", "-m", help="Embeddings model ID or tag")] = None,
        ) -> None:
            """
            Invoke an embedder.

            ex: uv run cli embedd "string to be embedded"
            """

            from rich.console import Console
            from rich.table import Table

            from genai_tk.core.embeddings_factory import EmbeddingsFactory

            factory = EmbeddingsFactory(
                embeddings=model,
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

        @cli_app.command(name="mcp-call")
        def mcp_call(
            server: Annotated[str, typer.Argument(help="MCP server name from config (mcpServers)")],
            server_args: Annotated[
                list[str] | None,
                typer.Argument(
                    help="Extra args appended to the server command args (e.g. an extra path for the MCP server)"
                ),
            ] = None,
            tool: Annotated[Optional[str], typer.Option("--tool", "-t", help="Tool name to call")] = None,
            tool_args: Annotated[
                Optional[str],
                typer.Option("--tool-args", help='JSON dict of arguments for the tool, e.g. \'{"path": "/tmp"}\''),
            ] = None,
        ) -> None:
            """Connect to an MCP server and list its tools, or call a specific tool.

            When called without ``--tool``, connects to the server and prints a table of
            all available tools with their descriptions.  With ``--tool``, calls that
            tool with the optional ``--tool-args`` JSON payload and prints the result.

            If ``server_args`` are provided they are *appended* to the ``args``
            list in the server config – useful to add extra allowed paths to the
            MCP server.

            Examples:
                cli core mcp-call filesystem
                cli core mcp-call filesystem /home/tcl/prj
                cli core mcp-call filesystem /home/tcl/prj --tool list_directory --tool-args '{"path": "."}'
                cli core mcp-call tavily-mcp --tool tavily-search --tool-args '{"query": "LangChain news"}'
            """
            import asyncio
            import json

            from rich.console import Console
            from rich.table import Table

            console = Console()

            async def _run() -> None:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

                from genai_tk.core.mcp_client import get_mcp_servers_dict

                try:
                    servers = get_mcp_servers_dict(filter=[server])
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    raise typer.Exit(1) from exc

                server_params_dict = servers[server]

                if server_args:
                    server_params_dict = dict(server_params_dict)
                    server_params_dict["args"] = list(server_params_dict.get("args", [])) + list(server_args)

                params = StdioServerParameters(**server_params_dict)

                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        if tool is None:
                            tools_response = await session.list_tools()
                            tbl = Table(
                                title=f"Tools – [bold cyan]{server}[/bold cyan]",
                                show_header=True,
                                header_style="bold magenta",
                            )
                            tbl.add_column("Tool", style="cyan", no_wrap=True)
                            tbl.add_column("Description")
                            for t in tools_response.tools:
                                tbl.add_row(t.name, t.description or "")
                            console.print(tbl)
                        else:
                            kwargs: dict = {}
                            if tool_args:
                                try:
                                    kwargs = json.loads(tool_args)
                                except json.JSONDecodeError as exc:
                                    console.print(f"[red]--tool-args is not valid JSON: {exc}[/red]")
                                    raise typer.Exit(1) from exc

                            console.print(
                                f"Calling [bold cyan]{server}[/bold cyan] → [bold yellow]{tool}[/bold yellow]"
                                + (f" with {kwargs}" if kwargs else "")
                                + " …"
                            )
                            result = await session.call_tool(tool, kwargs)
                            for content in result.content:
                                if hasattr(content, "text"):
                                    console.print(content.text)  # pyright: ignore[reportAttributeAccessIssue]
                                else:
                                    console.print(content)

            asyncio.run(_run())

        @cli_app.command()
        def similarity(
            sentences: Annotated[list[str], typer.Argument(help="List of sentences to compare (first is reference)")],
            model: Annotated[Optional[str], Option("--model", "-m", help="Embeddings model ID")] = None,
        ) -> None:
            """
            Calculate semantic similarity between sentences using cosine similarity.

            The first sentence is used as reference and compared to the others.

            Example:
                uv run cli core similarity "This is a test" "This is another test" "Completely different"
            """
            from langchain_community.utils.math import cosine_similarity

            from genai_tk.core.embeddings_factory import EmbeddingsFactory, get_embeddings

            if len(sentences) < 2:
                print("Error: At least 2 sentences are required")
                return

            if model is not None:
                if model not in EmbeddingsFactory.known_items():
                    print(f"Error: {model} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                    return

            embedder = get_embeddings(embeddings=model)

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
