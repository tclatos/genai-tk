"""CLI commands for RAG (Retrieval Augmented Generation) operations.

This module provides command-line interface commands for:
- Managing vector stores (create, delete, query)
- Adding documents to vector stores with embeddings
- Querying vector stores for similar documents
- Getting information and statistics about vector stores

The commands support multiple vector store backends configured via YAML
and provide rich formatted output with tables and panels.
"""

import json
import sys
from typing import Annotated, Optional

import typer
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from genai_tk.core.embeddings_store import EmbeddingsStore
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel


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


def _create_documents_table(documents: list[Document], show_scores: bool = False) -> Table:
    """Create a Rich table for displaying documents.

    Args:
        documents: List of documents to display
        show_scores: Whether to show relevance scores

    Returns:
        Rich Table with formatted document information
    """
    table = Table(title="Query Results", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Content", style="white")
    table.add_column("Metadata", style="dim white")

    if show_scores:
        table.add_column("Score", style="green", width=8)

    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        metadata = json.dumps(doc.metadata) if doc.metadata else "{}"

        if show_scores:
            # Note: scores would need to be passed separately in real implementation
            table.add_row(str(i), content, metadata, "N/A")
        else:
            table.add_row(str(i), content, metadata)

    return table


def _get_vector_store_safe(store_name: str) -> EmbeddingsStore | None:
    """Safely get a vector store by name, with error handling.

    Args:
        store_name: Name of the vector store configuration

    Returns:
        EmbeddingsStore instance or None if not found
    """
    console = Console()

    try:
        return EmbeddingsStore.create_from_config(store_name)
    except ValueError:
        available_configs = EmbeddingsStore.list_available_configs()
        if available_configs:
            config_list = ", ".join(f"'{config}'" for config in available_configs)
            error_msg = f"Vector store '{store_name}' not found.\nAvailable configurations: {config_list}"
        else:
            error_msg = f"Vector store '{store_name}' not found and no configurations are available."

        console.print(create_error_panel("Configuration Not Found", error_msg))
        return None
    except Exception as e:
        console.print(create_error_panel("Error", f"Failed to create vector store: {e}"))
        return None


class RagCommands(CliTopCommand):
    description: str = "Vector store operations for RAG (Retrieval Augmented Generation"

    def get_description(self) -> tuple[str, str]:
        return "rag", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def delete(store_name: Annotated[str, typer.Argument(help="Name of the vector store configuration")]) -> None:
            """Delete all documents from a vector store.

            This command clears all documents from the specified vector store while keeping
            the store configuration intact.
            """
            console = Console()

            vector_store = _get_vector_store_safe(store_name)
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
            store_name: Annotated[str, typer.Argument(help="Name of the vector store configuration")],
            text: Annotated[
                Optional[str], typer.Option("--text", "-t", help="Text to embed (or read from stdin)")
            ] = None,
            metadata: Annotated[
                Optional[str], typer.Option("--metadata", "-m", help="JSON metadata to attach to document")
            ] = None,
        ) -> None:
            """Embed text and store it in a vector store.

            The text can be provided via the --text option or read from stdin.
            Optional metadata can be provided as a JSON string.
            """
            console = Console()

            vector_store = _get_vector_store_safe(store_name)
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
            store_name: Annotated[str, typer.Argument(help="Name of the vector store configuration")],
            query_text: Annotated[str, typer.Argument(help="Query text to search for")],
            k: Annotated[int, typer.Option("--k", help="Number of results to return")] = 4,
            threshold: Annotated[
                Optional[float], typer.Option("--threshold", help="Minimum similarity score (0.0-1.0)")
            ] = None,
        ) -> None:
            """Query a vector store for similar documents.

            Search for documents similar to the provided query text and display
            the results in a formatted table.
            """
            console = Console()

            vector_store = _get_vector_store_safe(store_name)
            if not vector_store:
                return

            if k < 1:
                console.print(create_error_panel("Invalid Parameter", "k must be at least 1"))
                return

            if threshold is not None and (threshold < 0.0 or threshold > 1.0):
                console.print(create_error_panel("Invalid Parameter", "threshold must be between 0.0 and 1.0"))
                return

            try:
                # Perform the query
                results = vector_store.query(query_text, k=k, score_threshold=threshold)

                if not results:
                    message = f"No results found for query: '{query_text}'"
                    if threshold is not None:
                        message += f" (threshold: {threshold})"
                    console.print(create_warning_panel("No Results", message))
                    return

                # Display results
                console.print(_create_documents_table(results))

                # Show query summary
                summary_table = Table(title="Query Summary", show_header=True, header_style="bold blue")
                summary_table.add_column("Property", style="cyan")
                summary_table.add_column("Value", style="white")

                summary_table.add_row("Query", query_text)
                summary_table.add_row("Results Found", str(len(results)))
                summary_table.add_row("Requested (k)", str(k))
                if threshold is not None:
                    summary_table.add_row("Score Threshold", str(threshold))

                console.print(summary_table)

            except Exception as e:
                console.print(create_error_panel("Query Failed", f"Failed to query vector store: {e}"))

        @cli_app.command()
        def info(store_name: Annotated[str, typer.Argument(help="Name of the vector store configuration")]) -> None:
            """Get information and statistics about a vector store.

            Display detailed information about the vector store configuration,
            document count, and other relevant statistics.
            """
            console = Console()

            vector_store = _get_vector_store_safe(store_name)
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
