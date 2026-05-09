"""Prefect-powered file ingestion for RAG vector stores.

Typical usage::

    uv run cli workflow run rag_ingest \\
        --pathspec '**/*.md' --to my_retriever
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from prefect import flow, task
from upath import UPath

from genai_tk.core.factories.chunker_factory import ChunkerFactory
from genai_tk.core.factories.retriever_factory import ManagedRetriever, RetrieverFactory
from genai_tk.utils.file_patterns import resolve_files
from genai_tk.utils.hashing import file_digest


@dataclass(slots=True)
class FileToProcess:
    """File to be processed for RAG ingestion."""

    path: UPath
    content_hash: str
    content: str


def _load_file_content(path: UPath) -> str:
    """Load file content with error handling."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Error reading {}: {}", path, e)
        raise


def _prepare_files(
    files: list[UPath],
    force: bool,
    managed: ManagedRetriever,
) -> tuple[list[FileToProcess], int]:
    """Prepare files for processing, filtering out already processed files unless force is True.

    Args:
        files: List of file paths to process
        force: If True, process all files regardless of existing hashes
        managed: ManagedRetriever used to check for existing file hashes (Chroma only)

    Returns:
        Tuple of (files_to_process, skipped_count)
    """
    to_process: list[FileToProcess] = []
    skipped = 0

    # Attempt fast hash-based dedup for Chroma-backed vector stores
    existing_hashes: set[str] = set()
    vs = managed._vector_store
    if not force and vs is not None and hasattr(vs, "_collection"):
        try:
            all_docs = vs._collection.get(include=["metadatas"])  # type: ignore[attr-defined]
            if all_docs and "metadatas" in all_docs:
                for metadata in all_docs["metadatas"]:
                    if metadata and "file_hash" in metadata:
                        existing_hashes.add(metadata["file_hash"])
            logger.debug("Found {} existing file hashes in vector store", len(existing_hashes))
        except Exception as exc:
            logger.warning("Could not retrieve existing file hashes: {}", exc)

    for path in files:
        try:
            # Compute file hash
            content_hash = file_digest(path)

            # Skip if already processed (unless force is True)
            if not force and content_hash in existing_hashes:
                logger.info("Skipping already processed file: {}", path)
                skipped += 1
                continue

            # Load content
            content = _load_file_content(path)

            to_process.append(
                FileToProcess(
                    path=path,
                    content_hash=content_hash,
                    content=content,
                )
            )
        except Exception as e:
            logger.error("Error preparing {}: {}", path, e)
            continue

    return to_process, skipped


@task
def process_file_task(
    file_info: FileToProcess,
    retriever_name: str,
    max_chunk_tokens: int,
    chunker_name: str = "auto",
    root_dir: UPath | None = None,
) -> int:
    """Process a single file and add its chunks to the retriever store.

    Args:
        file_info: File information including path, hash, and content
        retriever_name: Name of the retriever configuration
        max_chunk_tokens: Maximum token count per chunk (used if chunker_name specifies size)
        chunker_name: Chunker configuration name ("auto" detects by file extension)
        root_dir: Root directory for computing relative paths

    Returns:
        Number of chunks added to the retriever store
    """
    logger.info("Processing file: {} (chunker: {})", file_info.path, chunker_name)

    try:
        # Get the appropriate chunker (auto-select based on file extension if needed)
        splitter = ChunkerFactory.create_for_file(file_info.path, chunker_name=chunker_name)
    except KeyError as exc:
        logger.error("Chunker configuration error for {}: {}", file_info.path, exc)
        raise

    # Create a Document from the file content and chunk it
    try:
        relative_path = file_info.path.relative_to(root_dir) if root_dir else file_info.path
        file_name = str(relative_path)
    except ValueError:
        file_name = str(file_info.path)

    # Split the document and create chunks with metadata
    base_metadata = {
        "source": file_name,
        "file_hash": file_info.content_hash,
    }

    documents = splitter.create_documents(
        [file_info.content],
        metadatas=[base_metadata],
    )

    if not documents:
        logger.warning("No chunks created for {}", file_info.path)
        return 0

    # Add chunk index to metadata
    for i, doc in enumerate(documents):
        doc.metadata["chunk_index"] = i
        doc.metadata["total_chunks"] = len(documents)

    managed = RetrieverFactory.create(retriever_name)
    managed.add_documents(documents)

    logger.info("Added {} chunks from {} to retriever '{}'", len(documents), file_info.path, retriever_name)
    return len(documents)


@flow(
    name="RAG File Ingestion",
    description="Ingest files into a RAG retriever store with parallel processing",
)
def rag_file_ingestion_flow(
    base_dir: str,
    retriever_name: str,
    max_chunk_tokens: int,
    chunker_name: str = "auto",
    pathspecs: list[str] | None = None,
    force: bool = False,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Ingest files into a RAG retriever store with parallel processing.

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        retriever_name: Name of the retriever configuration.
        max_chunk_tokens: Maximum token count per chunk.
        chunker_name: Chunker configuration name (``"auto"`` = detect by extension).
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).  Defaults to
            ``["**/*"]`` (all files, recursive).
        force: Reprocess all files regardless of existing hashes.
        batch_size: Number of files to process in parallel.

    Returns:
        Dictionary with ingestion statistics.
    """
    logger.info("Starting RAG file ingestion from '{}' to retriever '{}'", base_dir, retriever_name)

    files = resolve_files(base_dir, pathspecs=pathspecs)

    logger.info("Found {} files matching patterns", len(files))

    if not files:
        return {"total_files": 0, "processed_files": 0, "skipped_files": 0, "total_chunks": 0}

    managed = RetrieverFactory.create(retriever_name)
    files_to_process, skipped = _prepare_files(files, force, managed)

    logger.info("Processing {} files, skipping {} already processed files", len(files_to_process), skipped)

    if not files_to_process:
        return {"total_files": len(files), "processed_files": 0, "skipped_files": skipped, "total_chunks": 0}

    total_chunks = 0
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info("Processing batch {} ({} files)", i // batch_size + 1, len(batch))

        futures = [
            process_file_task.submit(
                file_info=fi,
                retriever_name=retriever_name,
                max_chunk_tokens=max_chunk_tokens,
                chunker_name=chunker_name,
                root_dir=None,
            )
            for fi in batch
        ]
        for future in futures:
            try:
                total_chunks += future.result()
            except Exception as exc:
                logger.error("Error processing file: {}", exc)

    logger.info("Completed: {} files, {} chunks", len(files_to_process), total_chunks)
    return {
        "total_files": len(files),
        "processed_files": len(files_to_process),
        "skipped_files": skipped,
        "total_chunks": total_chunks,
    }
