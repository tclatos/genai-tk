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
from typing import Annotated, Optional

import typer
from typer import Option

from genai_tk.cli.base import CliTopCommand
from genai_tk.core.cache import CacheMethod


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
            llm: Annotated[str, Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = "default",
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

            from genai_tk.agents.langchain_setup import setup_langchain
            from genai_tk.core.factories.llm_factory import LlmFactory

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
                llm=llm_id,  # use already-resolved ID to avoid duplicate resolution warnings
                json_mode=False,
                streaming=stream,
                reasoning=reasoning,
                cache=cache,
                llm_params={"temperature": temperature},
            )
            try:
                llm_model = llm_factory.get()
            except Exception as e:
                from rich.console import Console
                from rich.panel import Panel

                Console().print(
                    Panel(
                        f"[bold]{type(e).__name__}[/bold]\n{e}\n\n"
                        f"[dim]Model: [cyan]{llm_factory.info.model}[/cyan]   "
                        f"Provider: [cyan]{llm_factory.info.provider}[/cyan][/dim]",
                        title="[red]Model Init Error[/red]",
                        border_style="red",
                        expand=False,
                    )
                )
                return

            try:
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
            except Exception as e:
                from rich.console import Console
                from rich.panel import Panel

                err_type = type(e).__name__
                msg = str(e)
                # Extract the most useful line from httpx/openai errors
                for line in reversed(msg.splitlines()):
                    line = line.strip()
                    if line and not line.startswith("openai.") and "Error" not in line[:6]:
                        msg = line
                        break
                model_used = llm_factory.info.model if hasattr(llm_factory, "info") else (llm or "default")
                provider_used = llm_factory.info.provider if hasattr(llm_factory, "info") else ""
                Console().print(
                    Panel(
                        f"[bold]{err_type}[/bold]\n{msg}\n\n"
                        f"[dim]Model: [cyan]{model_used}[/cyan]   Provider: [cyan]{provider_used}[/cyan][/dim]\n"
                        f"[dim]Tip: run [bold]uv run cli info llm-profile {llm or 'MODEL_ID'}[/bold] to inspect the model[/dim]",
                        title="[red]API Error[/red]",
                        border_style="red",
                        expand=False,
                    )
                )

        @cli_app.command()
        def embedd(
            input: Annotated[str, typer.Argument(help="Text to embed")],
            model: Annotated[str, Option("--model", "-m", help="Embeddings model ID or tag")] = "default",
        ) -> None:
            """
            Invoke an embedder.

            ex: uv run cli embedd "string to be embedded"
            """

            from rich.console import Console
            from rich.table import Table

            from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory

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
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    raise typer.Exit(code=2) from exc

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
            model: Annotated[str, Option("--model", "-m", help="Embeddings model ID or tag")] = "default",
        ) -> None:
            """
            Calculate semantic similarity between sentences using cosine similarity.

            The first sentence is used as reference and compared to the others.

            Example:
                uv run cli core similarity "This is a test" "This is another test" "Completely different"
            """
            from langchain_community.utils.math import cosine_similarity

            from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory, get_embeddings

            if len(sentences) < 2:
                print("Error: At least 2 sentences are required")
                return

            resolved_id, error_msg = EmbeddingsFactory.resolve_embeddings_identifier_safe(model)
            if error_msg:
                print(error_msg)
                return

            embedder = get_embeddings(embeddings=resolved_id or "default")

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
