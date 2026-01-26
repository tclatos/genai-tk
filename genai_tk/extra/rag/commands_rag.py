"""CLI commands for RAG (Retrieval Augmented Generation) operations.

This module provides command-line interface commands for:
- Managing vector stores (create, delete, query)
- Adding documents to vector stores with embeddings
- Ingesting files into vector stores with parallel processing
- Querying vector stores for similar documents
- Getting information and statistics about vector stores

The commands support multiple vector store backends configured via YAML
and provide rich formatted output with tables and panels.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from genai_tk.main.cli import CliTopCommand

if TYPE_CHECKING:
    from genai_tk.core.embeddings_store import EmbeddingsStore


def _create_info_table(stats: dict) -> Table:
    """Create a Rich table for vector store information.

    Args:
        stats: Dictionary containing vector store statistics

    Returns:
        Rich Table with formatted statistics
    """
    table = Table(title="Vector Store Information", show_header=True, header_style="bold blue")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in stats.items():
        # Format the key to be more readable
        display_key = key.replace("_", " ").title()
        table.add_row(display_key, str(value))

    return table


def _create_documents_table(documents: list[Document], show_scores: bool = False, max_length: int = 100) -> Table:
    """Create a Rich table for displaying documents.

    Args:
        documents: List of documents to display
        show_scores: Whether to show relevance scores
        max_length: Maximum length of content to display per document

    Returns:
        Rich Table with formatted document information
    """
    table = Table(title="Query Results", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Content", style="white")
    table.add_column("Metadata", style="dim white")

    if show_scores:
        table.add_column("Score", style="green", width=8)

    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:max_length] + "..." if len(doc.page_content) > max_length else doc.page_content
        metadata = json.dumps(doc.metadata) if doc.metadata else "{}"

        if show_scores:
            # Note: scores would need to be passed separately in real implementation
            table.add_row(str(i), content, metadata, "N/A")
        else:
            table.add_row(str(i), content, metadata)

    return table


def _get_embeddings_store_safe(store_name: str) -> EmbeddingsStore | None:
    """Safely get a vector store by name, with error handling.

    Args:
        store_name: Name of the vector store configuration

    Returns:
        EmbeddingsStore instance or None if not found
    """

    from genai_tk.core.embeddings_store import EmbeddingsStore
    from genai_tk.utils.rich_widgets import create_error_panel

    console = Console()

    try:
        return EmbeddingsStore.create_from_config(store_name)
    except ValueError:
        available_configs = EmbeddingsStore.list_available_configs()
        if available_configs:
            # Create a nice table of available configurations
            table = Table(title="Available Vector Store Configurations", show_header=True, header_style="bold cyan")
            table.add_column("Configuration Name", style="green")

            for config in sorted(available_configs):
                table.add_row(config)

            console.print(
                create_error_panel(
                    "Configuration Not Found",
                    f"Vector store configuration '{store_name}' not found.\n\n"
                    f"Please choose from the available configurations below:",
                )
            )
            console.print(table)
            console.print("\n[dim]Tip: Use 'cli rag list-configs' to see detailed configuration information.[/dim]\n")
        else:
            console.print(
                create_error_panel(
                    "No Configurations Available",
                    f"Vector store '{store_name}' not found and no configurations are available.\n"
                    f"Please check your configuration file.",
                )
            )
        return None
    except Exception as e:
        console.print(create_error_panel("Error", f"Failed to create vector store: {e}"))
        return None


class RagCommands(CliTopCommand):
    description: str = "Vector store operations for RAG (Retrieval Augmented Generation"

    def get_description(self) -> tuple[str, str]:
        return "rag", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("add-files")
        def add_files(
            root_dir: Annotated[str, typer.Argument(help="Root directory containing files to ingest")],
            store_name: Annotated[
                str, typer.Option("--store", "-s", help="Name of the vector store configuration")
            ] = "default",
            include: Annotated[
                Optional[list[str]],
                typer.Option("--include", "-i", help="Include patterns (can be specified multiple times)"),
            ] = None,
            exclude: Annotated[
                Optional[list[str]],
                typer.Option("--exclude", "-e", help="Exclude patterns (can be specified multiple times)"),
            ] = None,
            recursive: Annotated[
                bool, typer.Option("--recursive/--no-recursive", "-r", help="Search recursively")
            ] = True,
            force: Annotated[bool, typer.Option("--force", "-f", help="Reprocess all files, ignoring hashes")] = False,
            batch_size: Annotated[
                int, typer.Option("--batch-size", "-b", help="Number of files to process in parallel")
            ] = 10,
            chunk_size: Annotated[int, typer.Option("--chunk-size", help="Maximum size of each chunk")] = 2000,
        ) -> None:
            """Ingest files into a vector store with parallel processing.

            Process files from a root directory using glob patterns, chunk them,
            and add them to a vector store. Uses file hashes to avoid reprocessing
            unchanged files unless --force is specified.

            For Markdown files, uses header-aware chunking; for other files, uses
            recursive character splitting.

            Examples:
                ```bash
                # Ingest all files in a directory
                cli rag add-files ./docs --store chroma_indexed

                # Ingest with custom patterns
                cli rag add-files ./docs --store chroma_indexed \\
                    --include "*.md" --include "*.txt" \\
                    --exclude "*draft*" --recursive

                # Force reprocessing of all files
                cli rag add-files ./docs --store chroma_indexed --force

                # Custom chunking parameters
                cli rag add-files ./docs --store chroma_indexed --chunk-size 3000

                # Process with larger batch size for better parallelism
                cli rag add-files ./docs --store chroma_indexed \\
                    --batch-size 20

                # Use config variables
                cli rag add-files '${paths.data_root}/docs' --store chroma_indexed
                ```
            """
            console = Console()

            # Resolve YAML config variables in path
            from genai_tk.utils.file_patterns import resolve_config_path
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel

            resolved_root_dir = resolve_config_path(root_dir)

            # Validate root directory
            root_path = Path(resolved_root_dir)
            if not root_path.exists():
                console.print(
                    create_error_panel(
                        "Invalid Path",
                        f"Root directory does not exist: {root_dir}"
                        + (f"\n(Resolved to: {resolved_root_dir})" if resolved_root_dir != root_dir else ""),
                    )
                )
                raise typer.Exit(1)

            if not root_path.is_dir():
                console.print(
                    create_error_panel(
                        "Invalid Path",
                        f"Path is not a directory: {root_dir}"
                        + (f"\n(Resolved to: {resolved_root_dir})" if resolved_root_dir != root_dir else ""),
                    )
                )
                raise typer.Exit(1)

            # Validate vector store
            vector_store = _get_embeddings_store_safe(store_name)
            if not vector_store:
                raise typer.Exit(1)

            # Default include patterns
            if include is None:
                include = ["**/*"]

            # Log the operation (show resolved path if different)
            display_path = (
                f"'{root_dir}'"
                if resolved_root_dir == root_dir
                else f"'{root_dir}' (resolved to '{resolved_root_dir}')"
            )
            console.print(f"[bold cyan]Starting file ingestion from {display_path} to store '{store_name}'[/bold cyan]")
            if include:
                console.print(f"  Include patterns: {', '.join(include)}")
            if exclude:
                console.print(f"  Exclude patterns: {', '.join(exclude)}")
            console.print(f"  Recursive: {recursive}")
            console.print(f"  Force: {force}")
            console.print(f"  Batch size: {batch_size}")
            console.print(f"  Chunk size: {chunk_size}")

            try:
                from genai_tk.extra.prefect.runtime import run_flow_ephemeral
                from genai_tk.extra.rag.rag_prefect_flow import rag_file_ingestion_flow

                result = run_flow_ephemeral(
                    rag_file_ingestion_flow,
                    root_dir=resolved_root_dir,
                    store_name=store_name,
                    include_patterns=include,
                    exclude_patterns=exclude,
                    recursive=recursive,
                    force=force,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                )

                # Display results
                result_table = Table(title="Ingestion Results", show_header=True, header_style="bold green")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="white")

                result_table.add_row("Total Files Found", str(result["total_files"]))
                result_table.add_row("Files Processed", str(result["processed_files"]))
                result_table.add_row("Files Skipped", str(result["skipped_files"]))
                result_table.add_row("Total Chunks Created", str(result["total_chunks"]))

                console.print(result_table)

                if result["processed_files"] > 0:
                    console.print(
                        create_success_panel(
                            "Ingestion Complete",
                            f"Successfully ingested {result['processed_files']} files "
                            f"({result['total_chunks']} chunks) into vector store '{store_name}'",
                        )
                    )
                else:
                    console.print(
                        create_warning_panel(
                            "No New Files",
                            "No new files were processed. Use --force to reprocess existing files.",
                        )
                    )

            except Exception as e:
                console.print(create_error_panel("Ingestion Failed", f"Failed to ingest files: {e}"))
                raise typer.Exit(1) from e

        @cli_app.command()
        def delete(
            store_name: Annotated[
                str, typer.Option("--store", "-s", help="Name of the vector store configuration")
            ] = "default",
            force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt")] = False,
        ) -> None:
            """Delete all documents from a vector store.

            This command clears all documents from the specified vector store while keeping
            the store configuration intact. Requires confirmation unless --force is used.
            """
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel

            console = Console()

            vector_store = _get_embeddings_store_safe(store_name)
            if not vector_store:
                return

            try:
                # Check if store has any documents first
                stats = vector_store.get_stats()
                doc_count = stats.get("document_count", "unknown")

                if doc_count == 0:
                    console.print(
                        create_warning_panel("Nothing to Delete", f"Vector store '{store_name}' is already empty.")
                    )
                    return

                # Ask for confirmation unless --force is used
                if not force:
                    if isinstance(doc_count, int):
                        message = (
                            f"Are you sure you want to delete {doc_count} documents from vector store '{store_name}'?"
                        )
                    else:
                        message = f"Are you sure you want to clear vector store '{store_name}'?"

                    confirmed = typer.confirm(message, default=False)
                    if not confirmed:
                        console.print(create_warning_panel("Cancelled", "Deletion cancelled by user."))
                        return

                success = vector_store.clear()
                if success:
                    if isinstance(doc_count, int):
                        message = f"Successfully deleted {doc_count} documents from vector store '{store_name}'"
                    else:
                        message = f"Successfully cleared vector store '{store_name}'"
                    console.print(create_success_panel("Deletion Complete", message))
                else:
                    console.print(
                        create_error_panel(
                            "Deletion Failed", f"Failed to clear vector store '{store_name}'. Check logs for details."
                        )
                    )
            except Exception as e:
                console.print(create_error_panel("Error", f"Failed to delete documents: {e}"))

        @cli_app.command()
        def embed(
            text: Annotated[
                Optional[str], typer.Option("--text", "-t", help="Text to embed (or read from stdin)")
            ] = None,
            metadata: Annotated[
                Optional[str], typer.Option("--metadata", "-m", help="JSON metadata to attach to document")
            ] = None,
            store_name: Annotated[
                str, typer.Option("--store", "-s", help="Name of the vector store configuration")
            ] = "default",
        ) -> None:
            """Embed text and store it in a vector store.

            The text can be provided via the --text option or read from stdin.
            Optional metadata can be provided as a JSON string.
            """
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel

            console = Console()

            vector_store = _get_embeddings_store_safe(store_name)
            if not vector_store:
                return

            # Get text from argument or stdin
            if text is None:
                if not sys.stdin.isatty():
                    text = sys.stdin.read().strip()
                else:
                    console.print(
                        create_error_panel("No Input", "Please provide text via --text option or pipe it through stdin")
                    )
                    return

            if not text:
                console.print(create_error_panel("Empty Input", "No text provided to embed"))
                return

            # Parse metadata if provided
            parsed_metadata = None
            if metadata:
                try:
                    parsed_metadata = json.loads(metadata)
                    if not isinstance(parsed_metadata, dict):
                        raise ValueError("Metadata must be a JSON object")
                except (json.JSONDecodeError, ValueError) as e:
                    console.print(create_error_panel("Invalid Metadata", f"Failed to parse metadata JSON: {e}"))
                    return

            try:
                # Embed the text
                vector_store.embed_text(text, parsed_metadata)

                # Show success message with stats
                stats = vector_store.get_stats()
                doc_count = stats.get("document_count", "unknown")

                message = f"Successfully embedded text into vector store '{store_name}'"
                if isinstance(doc_count, int):
                    message += f"\nTotal documents in store: {doc_count}"

                console.print(create_success_panel("Embedding Complete", message))

                # Show a preview of what was embedded
                preview_table = Table(title="Embedded Document", show_header=True, header_style="bold green")
                preview_table.add_column("Property", style="cyan")
                preview_table.add_column("Value", style="white")

                preview_text = text[:200] + "..." if len(text) > 200 else text
                preview_table.add_row("Text", preview_text)
                if parsed_metadata:
                    preview_table.add_row("Metadata", json.dumps(parsed_metadata, indent=2))

                console.print(preview_table)

            except Exception as e:
                console.print(create_error_panel("Embedding Failed", f"Failed to embed text: {e}"))

        @cli_app.command()
        def query(
            query_text: Annotated[str, typer.Argument(help="Query text to search for")],
            k: Annotated[int, typer.Option("--k", help="Number of results to return")] = 4,
            filter: Annotated[
                Optional[str],
                typer.Option("--filter", help='Metadata filter as JSON string (e.g., \'{"file_hash": "abc123"}\''),
            ] = None,
            full: Annotated[bool, typer.Option("--full", help="Show full document content")] = False,
            max_length: Annotated[
                int, typer.Option("--max-length", "-l", help="Maximum length of content to display per document")
            ] = 100,
            store_name: Annotated[
                str, typer.Option("--store", "-s", help="Name of the vector store configuration")
            ] = "default",
        ) -> None:
            """Query a vector store for similar documents.

            Search for documents similar to the provided query text and display
            the results in a formatted table. Optionally filter by metadata.

            Example:
                cli rag query "CNES" --filter '{"file_hash": "1fa730def69ff25e"}'
            """
            from genai_tk.utils.rich_widgets import create_error_panel, create_warning_panel

            console = Console()

            embeddings_store = _get_embeddings_store_safe(store_name)
            if not embeddings_store:
                return

            if k < 1:
                console.print(create_error_panel("Invalid Parameter", "k must be at least 1"))
                return

            # Parse metadata filter if provided
            metadata_filter = None
            if filter:
                try:
                    metadata_filter = json.loads(filter)
                except json.JSONDecodeError as e:
                    console.print(create_error_panel("Invalid Filter", f"Failed to parse filter JSON: {e}"))
                    return

            try:
                # Perform the query (async)
                results = asyncio.run(embeddings_store.query(query_text, k=k, filter=metadata_filter))

                if not results:
                    message = f"No results found for query: '{query_text}'"
                    if metadata_filter:
                        message += f" (filter: {json.dumps(metadata_filter)})"
                    console.print(create_warning_panel("No Results", message))
                    return

                # Display results with full content if requested
                if full:
                    for i, doc in enumerate(results, 1):
                        console.print(f"\n[bold cyan]Document {i}[/bold cyan]")
                        console.print(f"[bold]Content:[/bold]\n{doc.page_content}")
                        if doc.metadata:
                            console.print("\n[bold]Metadata:[/bold]")
                            for key, value in doc.metadata.items():
                                console.print(f"  {key}: {value}")
                        console.print("-" * 80)
                else:
                    console.print(_create_documents_table(results, max_length=max_length))

                # Show query summary
                summary_table = Table(title="Query Summary", show_header=True, header_style="bold blue")
                summary_table.add_column("Property", style="cyan")
                summary_table.add_column("Value", style="white")

                summary_table.add_row("Query", query_text)
                summary_table.add_row("Results Found", str(len(results)))
                summary_table.add_row("Requested (k)", str(k))
                if metadata_filter:
                    summary_table.add_row("Metadata Filter", json.dumps(metadata_filter))

                console.print(summary_table)

            except Exception as e:
                console.print(create_error_panel("Query Failed", f"Failed to query vector store: {e}"))
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        @cli_app.command()
        def info(
            store_name: Annotated[
                str, typer.Option("--store", "-s", help="Name of the vector store configuration")
            ] = "default",
        ) -> None:
            """Get information and statistics about a vector store.

            Display detailed information about the vector store configuration,
            document count, and other relevant statistics.
            """
            from genai_tk.utils.rich_widgets import create_error_panel

            console = Console()

            vector_store = _get_embeddings_store_safe(store_name)
            if not vector_store:
                return

            try:
                stats = vector_store.get_stats()
                console.print(_create_info_table(stats))

            except Exception as e:
                console.print(create_error_panel("Info Failed", f"Failed to get vector store info: {e}"))

        @cli_app.command()
        def list_configs() -> None:
            """List all available vector store configurations.

            Display all vector store configurations that can be used with the
            other RAG commands, including backend type and storage information.
            """
            console = Console()
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.utils.rich_widgets import create_error_panel, create_warning_panel

            try:
                configs = EmbeddingsStore.list_available_configs()

                if not configs:
                    console.print(
                        create_warning_panel(
                            "No Configurations",
                            "No vector store configurations found. Check your YAML configuration file.",
                        )
                    )
                    return

                table = Table(title="Available Vector Store Configurations", show_header=True, header_style="bold cyan")
                table.add_column("Configuration Name", style="green", width=20)
                table.add_column("Backend", style="blue", width=10)
                table.add_column("Storage", style="yellow", width=11)
                table.add_column("Embeddings", style="magenta", width=15)
                table.add_column("Usage Example", style="dim white")

                for config in sorted(configs):
                    # Try to get detailed info about each config
                    backend = "Unknown"
                    storage = "Unknown"
                    embeddings = "Unknown"

                    try:
                        # Safely create the vector store to get its info
                        store = EmbeddingsStore.create_from_config(config)
                        backend = store.backend or "Unknown"

                        # Get embeddings information
                        embeddings_id = store.embeddings_factory.embeddings_id or "default"
                        # Simplify the embeddings ID for display
                        if "_" in embeddings_id:
                            # Extract meaningful part (e.g., "ada_002_openai" -> "ada_002")
                            parts = embeddings_id.split("_")
                            if len(parts) >= 2:
                                embeddings = "_".join(parts[:2])  # Take first two parts
                            else:
                                embeddings = embeddings_id
                        else:
                            embeddings = embeddings_id

                        # Determine storage type
                        if backend == "Chroma":
                            storage_config = store.config.get("storage", "::memory::")
                            storage = "Memory" if storage_config == "::memory::" else "Persistent"
                        elif backend == "InMemory":
                            storage = "Memory"
                        elif backend == "Sklearn":
                            storage = "Memory"
                        elif backend == "PgVector":
                            storage = "Database"
                        else:
                            storage = "Unknown"

                    except Exception:
                        # If we can't create the store, show what we can
                        backend = "Error"
                        storage = "Error"
                        embeddings = "Error"

                    table.add_row(config, backend, storage, embeddings, f"uv run cli rag info {config}")

                console.print(table)

            except Exception as e:
                console.print(create_error_panel("List Failed", f"Failed to list configurations: {e}"))
