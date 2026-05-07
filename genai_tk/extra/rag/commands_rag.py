"""CLI commands for RAG (Retrieval-Augmented Generation) operations.

Commands:
    add-files       Ingest files into a retriever store
    query           Search a retriever for similar documents
    embed           Embed a single text snippet into a retriever store
    delete          Delete/clear a retriever store
    info            Show retriever configuration and statistics
    list-retrievers List all available retriever configurations
    list-configs    List all available embeddings_store configurations (legacy)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand

if TYPE_CHECKING:
    from genai_tk.core.factories.retriever_factory import ManagedRetriever


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------


def _create_info_table(stats: dict) -> Table:
    table = Table(title="Retriever Information", show_header=True, header_style="bold blue")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    return table


def _create_documents_table(documents: list, max_length: int = 100) -> Table:
    table = Table(title="Query Results", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Content", style="white")
    table.add_column("Metadata", style="dim white")
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:max_length] + "..." if len(doc.page_content) > max_length else doc.page_content
        metadata = json.dumps(doc.metadata) if doc.metadata else "{}"
        table.add_row(str(i), content, metadata)
    return table


def _get_retriever_safe(retriever_name: str) -> "ManagedRetriever | None":
    from genai_tk.core.factories.retriever_factory import RetrieverFactory
    from genai_tk.utils.rich_widgets import create_error_panel

    console = Console()
    try:
        return RetrieverFactory.create(retriever_name)
    except ValueError:
        available = RetrieverFactory.list_available_configs()
        if available:
            table = Table(title="Available Retrievers", show_header=True, header_style="bold cyan")
            table.add_column("Retriever Name", style="green")
            for name in sorted(available):
                table.add_row(name)
            console.print(
                create_error_panel(
                    "Retriever Not Found",
                    f"Retriever '{retriever_name}' not found. Choose from the list below:",
                )
            )
            console.print(table)
        else:
            console.print(
                create_error_panel(
                    "No Retrievers Configured",
                    f"Retriever '{retriever_name}' not found and no retrievers are configured.\n"
                    "Add a 'retrievers:' section to your YAML configuration.",
                )
            )
        return None
    except Exception as exc:
        console.print(create_error_panel("Error", f"Failed to create retriever: {exc}"))
        return None


def _resolve_add_files_params(
    *,
    root_dir: str | None,
    retriever_name: str | None,
    include: list[str] | None,
    exclude: list[str] | None,
    recursive: bool | None,
    force: bool,
    batch_size: int | None,
    chunk_size: int | None,
    chunker: str | None,
    workflow_config: str | None,
    config: Any = None,
) -> tuple[dict[str, Any], Any | None]:
    """Resolve effective parameters for ``rag add-files``.

    When ``workflow_config`` is provided, values are loaded from a workflow or
    workflow profile and merged with explicit CLI overrides. Otherwise the
    legacy direct-CLI behavior is used.
    """
    if workflow_config:
        from genai_tk.workflow.resolver import WorkflowResolutionError, resolve_workflow_invocation

        cli_overrides: dict[str, Any] = {}
        if root_dir is not None:
            cli_overrides["root_dir"] = root_dir
        if retriever_name is not None:
            cli_overrides["retriever_name"] = retriever_name
        if include is not None:
            cli_overrides["include_patterns"] = include
        if exclude is not None:
            cli_overrides["exclude_patterns"] = exclude
        if recursive is not None:
            cli_overrides["recursive"] = recursive
        if batch_size is not None:
            cli_overrides["batch_size"] = batch_size
        if chunk_size is not None:
            cli_overrides["chunk_size"] = chunk_size
        if chunker is not None:
            cli_overrides["chunker"] = chunker

        invocation = resolve_workflow_invocation(
            workflow_config,
            cli_overrides=cli_overrides,
            config=config,
            force=force,
        )
        if invocation.workflow_name != "rag_add_files":
            raise WorkflowResolutionError(
                f"Workflow/profile '{workflow_config}' resolves to '{invocation.workflow_name}', expected 'rag_add_files'."
            )

        values = invocation.values
        effective_root_dir = values.get("root_dir")
        if not effective_root_dir:
            raise WorkflowResolutionError("Resolved rag_add_files invocation is missing required value 'root_dir'.")

        return (
            {
                "root_dir": str(effective_root_dir),
                "retriever_name": str(values.get("retriever_name", "default")),
                "include": values.get("include_patterns", ["**/*"]),
                "exclude": values.get("exclude_patterns"),
                "recursive": bool(values.get("recursive", True)),
                "force": force,
                "batch_size": int(values.get("batch_size", 10)),
                "chunk_size": int(values.get("chunk_size", 512)),
                "chunker": str(values.get("chunker", "auto")),
            },
            invocation,
        )

    if root_dir is None:
        raise ValueError("root_dir is required when --config/--profile is not used")

    return (
        {
            "root_dir": root_dir,
            "retriever_name": retriever_name or "default",
            "include": include or ["**/*"],
            "exclude": exclude,
            "recursive": True if recursive is None else recursive,
            "force": force,
            "batch_size": 10 if batch_size is None else batch_size,
            "chunk_size": 512 if chunk_size is None else chunk_size,
            "chunker": chunker or "auto",
        },
        None,
    )


def _print_add_files_resolution(console: Console, params: dict[str, Any], invocation: Any | None) -> None:
    """Render the effective ``rag add-files`` parameters."""
    table = Table(title="RAG Add-Files Resolution", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Workflow", getattr(invocation, "workflow_name", "<legacy>"))
    table.add_row("Profile", getattr(invocation, "profile_name", "<none>") or "<none>")
    table.add_row("Root Dir", str(params["root_dir"]))
    table.add_row("Retriever", str(params["retriever_name"]))
    table.add_row("Recursive", str(params["recursive"]))
    table.add_row("Batch Size", str(params["batch_size"]))
    table.add_row("Chunk Size", str(params["chunk_size"]))
    table.add_row("Chunker", str(params["chunker"]))
    table.add_row("Include", json.dumps(params["include"]))
    table.add_row("Exclude", json.dumps(params["exclude"]))
    console.print(table)


# ---------------------------------------------------------------------------
# Command class
# ---------------------------------------------------------------------------


class RagCommands(CliTopCommand):
    description: str = "RAG (Retrieval-Augmented Generation) operations"

    def get_description(self) -> tuple[str, str]:
        return "rag", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # noqa: C901, PLR0912, PLR0915

        # ------------------------------------------------------------------ #
        # add-files
        # ------------------------------------------------------------------ #
        @cli_app.command("add-files")
        def add_files(
            root_dir: Annotated[
                str | None,
                typer.Argument(help="Root directory containing files to ingest", show_default=False),
            ] = None,
            workflow_config: Annotated[
                str | None,
                typer.Option("--config", "--profile", help="Workflow profile or workflow name for YAML-driven mode"),
            ] = None,
            retriever_name: Annotated[
                str | None,
                typer.Option("--retriever", "-r", "--store", "-s", help="Retriever configuration name"),
            ] = None,
            include: Annotated[
                Optional[list[str]],
                typer.Option("--include", "-i", help="Include glob patterns (repeatable)"),
            ] = None,
            exclude: Annotated[
                Optional[list[str]],
                typer.Option("--exclude", "-e", help="Exclude glob patterns (repeatable)"),
            ] = None,
            recursive: Annotated[bool | None, typer.Option("--recursive/--no-recursive")] = None,
            force: Annotated[bool, typer.Option("--force", "-f", help="Reprocess all files")] = False,
            dry_run: Annotated[
                bool, typer.Option("--dry-run", help="Resolve parameters without executing ingestion")
            ] = False,
            batch_size: Annotated[int | None, typer.Option("--batch-size", "-b")] = None,
            chunk_size: Annotated[int | None, typer.Option("--chunk-size", help="Max chunk size in tokens")] = None,
            chunker: Annotated[
                str | None,
                typer.Option(
                    "--chunker",
                    help='Chunker name (e.g. "markdown", "recursive") or "auto" to detect by file extension',
                ),
            ] = None,
        ) -> None:
            """Ingest files from a directory into a retriever store."""
            from genai_tk.utils.file_patterns import resolve_config_path
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel
            from genai_tk.workflow.resolver import WorkflowResolutionError

            console = Console()

            try:
                params, invocation = _resolve_add_files_params(
                    root_dir=root_dir,
                    retriever_name=retriever_name,
                    include=include,
                    exclude=exclude,
                    recursive=recursive,
                    force=force,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    chunker=chunker,
                    workflow_config=workflow_config,
                )
            except (ValueError, WorkflowResolutionError) as exc:
                console.print(create_error_panel("Invalid Parameters", str(exc)))
                raise typer.Exit(1) from exc

            _print_add_files_resolution(console, params, invocation)
            if dry_run:
                console.print(create_success_panel("Dry Run", "Resolved parameters only; no ingestion was executed."))
                return

            resolved = resolve_config_path(params["root_dir"])
            root_path = Path(resolved)

            if not root_path.exists() or not root_path.is_dir():
                console.print(create_error_panel("Invalid Path", f"Directory not found: {params['root_dir']}"))
                raise typer.Exit(1)

            console.print(
                f"[bold cyan]Ingesting '{params['root_dir']}' → retriever '{params['retriever_name']}'[/bold cyan]"
            )

            try:
                from genai_tk.extra.prefect.runtime import run_flow_ephemeral
                from genai_tk.extra.rag.rag_prefect_flow import rag_file_ingestion_flow

                result = run_flow_ephemeral(
                    rag_file_ingestion_flow,
                    root_dir=resolved,
                    retriever_name=params["retriever_name"],
                    max_chunk_tokens=params["chunk_size"],
                    chunker_name=params["chunker"],
                    include_patterns=params["include"],
                    exclude_patterns=params["exclude"],
                    recursive=params["recursive"],
                    force=params["force"],
                    batch_size=params["batch_size"],
                )

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
                            f"Ingested {result['processed_files']} files ({result['total_chunks']} chunks) "
                            f"into retriever '{params['retriever_name']}'",
                        )
                    )
                else:
                    console.print(
                        create_warning_panel("No New Files", "No new files processed. Use --force to reprocess.")
                    )
            except Exception as exc:
                console.print(create_error_panel("Ingestion Failed", f"{exc}"))
                raise typer.Exit(1) from exc

        # ------------------------------------------------------------------ #
        # query
        # ------------------------------------------------------------------ #
        @cli_app.command()
        def query(
            query_text: Annotated[str, typer.Argument(help="Query text")] = "",
            retriever_name: Annotated[
                str, typer.Option("--retriever", "-r", "--store", "-s", help="Retriever configuration name")
            ] = "default",
            k: Annotated[int, typer.Option("--k", help="Number of results")] = 4,
            filter: Annotated[  # noqa: A002
                Optional[str],
                typer.Option("--filter", help='Metadata filter JSON, e.g. \'{"source": "docs"}\''),
            ] = None,
            full: Annotated[bool, typer.Option("--full", help="Show full document content")] = False,
            max_length: Annotated[int, typer.Option("--max-length", "-l")] = 100,
        ) -> None:
            """Search a retriever for documents similar to the query."""
            from genai_tk.utils.rich_widgets import create_error_panel, create_warning_panel

            console = Console()
            managed = _get_retriever_safe(retriever_name)
            if not managed:
                raise typer.Exit(1)

            if k < 1:
                console.print(create_error_panel("Invalid Parameter", "k must be at least 1"))
                raise typer.Exit(1)

            metadata_filter: dict | None = None
            if filter:
                try:
                    metadata_filter = json.loads(filter)
                except json.JSONDecodeError as exc:
                    console.print(create_error_panel("Invalid Filter", f"{exc}"))
                    raise typer.Exit(1) from exc

            try:
                results = asyncio.run(managed.aquery(query_text, k=k, filter=metadata_filter))
            except Exception as exc:
                console.print(create_error_panel("Query Failed", f"{exc}"))
                raise typer.Exit(1) from exc

            if not results:
                console.print(create_warning_panel("No Results", f"No results for: '{query_text}'"))
                return

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

            summary = Table(title="Query Summary", show_header=True, header_style="bold blue")
            summary.add_column("Property", style="cyan")
            summary.add_column("Value", style="white")
            summary.add_row("Retriever", retriever_name)
            summary.add_row("Query", query_text)
            summary.add_row("Results Found", str(len(results)))
            if metadata_filter:
                summary.add_row("Filter", json.dumps(metadata_filter))
            console.print(summary)

        # ------------------------------------------------------------------ #
        # embed
        # ------------------------------------------------------------------ #
        @cli_app.command()
        def embed(
            retriever_name: Annotated[str, typer.Argument(help="Retriever configuration name")] = "default",
            text: Annotated[Optional[str], typer.Option("--text", "-t")] = None,
            metadata: Annotated[Optional[str], typer.Option("--metadata", "-m", help="JSON metadata")] = None,
        ) -> None:
            """Embed a single text snippet and store it in the retriever."""
            from langchain_core.documents import Document

            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel

            console = Console()
            managed = _get_retriever_safe(retriever_name)
            if not managed:
                raise typer.Exit(1)

            if text is None:
                import click

                stdin = click.get_text_stream("stdin")
                text = stdin.read().strip() if not stdin.isatty() else None
            if not text:
                console.print(create_error_panel("No Input", "Provide text via --text or stdin"))
                raise typer.Exit(1)

            parsed_meta: dict | None = None
            if metadata:
                try:
                    parsed_meta = json.loads(metadata)
                except json.JSONDecodeError as exc:
                    console.print(create_error_panel("Invalid Metadata", f"{exc}"))
                    raise typer.Exit(1) from exc

            doc = Document(page_content=text, metadata=parsed_meta or {"source": "cli-embed"})
            try:
                asyncio.run(managed.aadd_documents([doc]))
                console.print(create_success_panel("Embedded", f"Text stored in retriever '{retriever_name}'"))
            except Exception as exc:
                console.print(create_error_panel("Failed", f"{exc}"))

        # ------------------------------------------------------------------ #
        # delete
        # ------------------------------------------------------------------ #
        @cli_app.command()
        def delete(
            retriever_name: Annotated[str, typer.Argument(help="Retriever configuration name")] = "default",
            force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
        ) -> None:
            """Delete/clear all stored documents for a retriever."""
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel

            console = Console()
            managed = _get_retriever_safe(retriever_name)
            if not managed:
                raise typer.Exit(1)

            if not force and not typer.confirm(
                f"Delete all stored documents for retriever '{retriever_name}'?", default=False
            ):
                console.print(create_warning_panel("Cancelled", "Deletion cancelled."))
                return

            try:
                success = asyncio.run(managed.adelete_store())
                if success:
                    console.print(create_success_panel("Deleted", f"Store for retriever '{retriever_name}' cleared."))
                else:
                    console.print(create_error_panel("Failed", "Could not delete store. Check logs."))
            except NotImplementedError as exc:
                console.print(create_error_panel("Not Supported", str(exc)))
            except Exception as exc:
                console.print(create_error_panel("Error", f"{exc}"))

        # ------------------------------------------------------------------ #
        # info
        # ------------------------------------------------------------------ #
        @cli_app.command()
        def info(
            retriever_name: Annotated[str, typer.Argument(help="Retriever configuration name")] = "default",
        ) -> None:
            """Show configuration and statistics for a retriever."""
            from genai_tk.utils.rich_widgets import create_error_panel

            console = Console()
            managed = _get_retriever_safe(retriever_name)
            if not managed:
                raise typer.Exit(1)
            try:
                console.print(_create_info_table(managed.get_stats()))
            except Exception as exc:
                console.print(create_error_panel("Error", f"{exc}"))

        # ------------------------------------------------------------------ #
        # list-retrievers
        # ------------------------------------------------------------------ #
        @cli_app.command("list-retrievers")
        def list_retrievers() -> None:
            """List all configured retrievers from the ``retrievers:`` YAML section."""
            from genai_tk.core.factories.retriever_factory import RetrieverFactory
            from genai_tk.utils.rich_widgets import create_warning_panel

            console = Console()
            configs = RetrieverFactory.list_available_configs()

            if not configs:
                console.print(
                    create_warning_panel(
                        "No Retrievers",
                        "No retriever configurations found. Add a 'retrievers:' section to your YAML config.",
                    )
                )
                return

            table = Table(title="Available Retrievers", show_header=True, header_style="bold cyan")
            table.add_column("Name", style="green", width=24)
            table.add_column("Type", style="blue", width=14)
            table.add_column("Details", style="yellow")

            for name in sorted(configs):
                try:
                    raw = _get_retriever_raw_config(name)
                    rtype = raw.get("type", "unknown")
                    details = _summarise_retriever_config(rtype, raw)
                except Exception:
                    rtype, details = "?", ""
                table.add_row(name, rtype, details)

            console.print(table)

        # ------------------------------------------------------------------ #
        # list-configs  (legacy: embeddings_store configs)
        # ------------------------------------------------------------------ #
        @cli_app.command("list-configs")
        def list_configs() -> None:
            """List embeddings_store configurations (legacy, use list-retrievers instead)."""
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.utils.rich_widgets import create_warning_panel

            console = Console()
            configs = EmbeddingsStore.list_available_configs()

            if not configs:
                console.print(create_warning_panel("No Configs", "No embeddings_store configurations found."))
                return

            table = Table(title="Embeddings Store Configurations", show_header=True, header_style="bold cyan")
            table.add_column("Name", style="green", width=24)
            table.add_column("Backend", style="blue", width=10)
            table.add_column("Storage", style="yellow", width=12)
            table.add_column("Embeddings", style="magenta")

            for name in sorted(configs):
                try:
                    store = EmbeddingsStore.create_from_config(name)
                    backend = store.backend or "?"
                    storage = "memory" if store.config.get("storage") == "::memory::" else "disk"
                    emb = (store.embeddings_factory.embeddings_id or "").split("@")[0]
                except Exception:
                    backend = storage = emb = "?"
                table.add_row(name, backend, storage, emb)

            console.print(table)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_retriever_raw_config(name: str) -> dict:
    from genai_tk.utils.config_mngr import global_config

    return global_config().get_dict(f"retrievers.{name}")


def _summarise_retriever_config(rtype: str, raw: dict) -> str:
    if rtype == "vector":
        return f"embeddings_store={raw.get('embeddings_store', '?')}, top_k={raw.get('top_k', 4)}"
    if rtype == "bm25":
        return f"preprocessing={raw.get('preprocessing', 'default')}, k={raw.get('k', 4)}"
    if rtype == "pg_hybrid":
        return f"postgres={raw.get('postgres', 'default')}, hybrid={raw.get('hybrid_search', True)}"
    if rtype == "ensemble":
        refs = [c.get("ref", "?") for c in raw.get("retrievers", [])]
        return "refs=[" + ", ".join(refs) + "]"
    if rtype == "reranked":
        return f"base={raw.get('retriever', '?')}, reranker={raw.get('reranker', 'embeddings')}"
    if rtype == "zero_entropy":
        return f"collection={raw.get('collection_name', '?')}"
    return ""


def register_commands(cli_app: typer.Typer) -> None:
    """Register the RAG command group onto *cli_app*.

    Convenience function used by integration tests and any caller that wants
    to mount the ``rag`` sub-commands without going through the full
    ``CliTopCommand`` discovery mechanism.

    Args:
        cli_app: Typer application to add the ``rag`` sub-app to.
    """
    RagCommands().register(cli_app)
