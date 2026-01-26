"""Prefect-powered file ingestion for RAG vector stores.

This module defines a Prefect flow that processes files, chunks them,
and adds them to a vector store with deduplication based on file hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chonkie import RecursiveChunker
from langchain_core.documents import Document
from loguru import logger
from prefect import flow, task
from upath import UPath

from genai_tk.core.embeddings_store import EmbeddingsStore
from genai_tk.extra.rag.markdown_chunking import chunk_markdown_content
from genai_tk.utils.file_patterns import resolve_files
from genai_tk.utils.hashing import file_digest

# Initialize text chunker once (it is reusable and thread-safe)
_text_chunker: RecursiveChunker | None = None


def _get_text_chunker(max_chunk_tokens: int) -> RecursiveChunker:
    """Get or create a text chunker."""
    global _text_chunker
    if _text_chunker is None or _text_chunker.chunk_size != max_chunk_tokens:
        _text_chunker = RecursiveChunker(
            chunk_size=max_chunk_tokens,
            tokenizer="character",
        )
    return _text_chunker


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
        logger.error(f"Error reading {path}: {e}")
        raise


def _chunk_markdown(content: str, max_chunk_tokens: int) -> list[str]:
    """Chunk markdown content using the markdown_chunking module.

    Uses chunk_markdown_content to parse markdown and extract tables, code blocks,
    and text chunks separately. Returns only the text content of each chunk.
    """

    chunks = chunk_markdown_content(content, max_tokens=max_chunk_tokens)
    return [chunk.content for chunk in chunks]


def _chunk_text(content: str, max_chunk_tokens: int) -> list[str]:
    """Chunk text content using chonkie's RecursiveChunker."""
    chunker = _get_text_chunker(max_chunk_tokens)
    chunks = chunker.chunk(content)
    return [chunk.text for chunk in chunks]


def _chunk_file_content(path: UPath, content: str, max_chunk_tokens: int) -> list[str]:
    """Chunk file content based on file type."""
    if path.suffix.lower() in {".md", ".markdown"}:
        logger.debug(f"Using MarkdownChef for {path}")
        return _chunk_markdown(content, max_chunk_tokens=max_chunk_tokens)
    else:
        logger.debug(f"Using text chunker for {path}")
        return _chunk_text(content, max_chunk_tokens=max_chunk_tokens)


def _prepare_files(
    files: list[UPath],
    force: bool,
    vector_store: EmbeddingsStore,
) -> tuple[list[FileToProcess], int]:
    """Prepare files for processing, filtering out already processed files unless force is True.

    Args:
        files: List of file paths to process
        force: If True, process all files regardless of existing hashes
        vector_store: Vector store instance to check for existing files

    Returns:
        Tuple of (files_to_process, skipped_count)
    """
    to_process: list[FileToProcess] = []
    skipped = 0

    # Get existing document metadata from vector store if available
    existing_hashes: set[str] = set()
    if not force and vector_store.backend == "Chroma":
        try:
            # Try to get existing file hashes from vector store metadata
            vs = vector_store.get_vector_store()
            all_docs = vs._collection.get(include=["metadatas"])  # type: ignore
            if all_docs and "metadatas" in all_docs:
                for metadata in all_docs["metadatas"]:
                    if metadata and "file_hash" in metadata:
                        existing_hashes.add(metadata["file_hash"])
            logger.debug(f"Found {len(existing_hashes)} existing file hashes in vector store")
        except Exception as e:
            logger.warning(f"Could not retrieve existing file hashes: {e}")

    for path in files:
        try:
            # Compute file hash
            content_hash = file_digest(path)

            # Skip if already processed (unless force is True)
            if not force and content_hash in existing_hashes:
                logger.info(f"Skipping already processed file: {path}")
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
            logger.error(f"Error preparing {path}: {e}")
            continue

    return to_process, skipped


@task
def process_file_task(
    file_info: FileToProcess,
    store_name: str,
    max_chunk_tokens: int,
    root_dir: UPath | None = None,
) -> int:
    """Process a single file and add its chunks to the vector store.

    Args:
        file_info: File information including path, hash, and content
        store_name: Name of the vector store configuration
        max_chunk_tokens: Maximum token count per chunk
        root_dir: Root directory for computing relative paths

    Returns:
        Number of chunks added to the vector store
    """
    logger.info(f"Processing file: {file_info.path}")

    # Chunk the content
    chunks = _chunk_file_content(
        file_info.path,
        file_info.content,
        max_chunk_tokens=max_chunk_tokens,
    )

    if not chunks:
        logger.warning(f"No chunks created for {file_info.path}")
        return 0

    # Compute relative path
    try:
        relative_path = file_info.path.relative_to(root_dir)
        file_name = str(relative_path)
    except ValueError:
        # If file is not under root_dir, use absolute path
        file_name = str(file_info.path)

    # Create documents with metadata
    documents = []
    for i, chunk_text in enumerate(chunks):
        metadata = {
            "source": file_name,
            "file_hash": file_info.content_hash,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        doc = Document(page_content=chunk_text, metadata=metadata)
        documents.append(doc)

    # Add to vector store
    vector_store = EmbeddingsStore.create_from_config(store_name)
    _ = vector_store.add_documents(documents)

    logger.info(f"Added {len(documents)} chunks from {file_info.path} to vector store")
    return len(documents)


@flow(
    name="RAG File Ingestion",
    description="Ingest files into RAG vector store with parallel processing",
)
def rag_file_ingestion_flow(
    root_dir: str,
    store_name: str,
    max_chunk_tokens: int,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    recursive: bool = True,
    force: bool = False,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Ingest files into a RAG vector store with parallel processing.

    Args:
        root_dir: Root directory containing files to process
        store_name: Name of the vector store configuration
        max_chunk_tokens: Maximum token count per chunk
        include_patterns: List of glob patterns for files to include
        exclude_patterns: List of glob patterns for files to exclude
        recursive: Whether to search directories recursively
        force: If True, reprocess all files regardless of existing hashes
        batch_size: Number of files to process in parallel

    Returns:
        Dictionary with statistics about the ingestion process
    """
    logger.info(f"Starting RAG file ingestion from '{root_dir}' to store '{store_name}'")

    # Resolve files
    root_path = UPath(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    if include_patterns is None:
        include_patterns = ["**/*"]

    files = resolve_files(
        str(root_path),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        recursive=recursive,
    )

    logger.info(f"Found {len(files)} files matching patterns")

    if not files:
        logger.warning("No files found to process")
        return {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "total_chunks": 0,
        }

    # Get vector store instance for checking existing files
    vector_store = EmbeddingsStore.create_from_config(store_name)

    # Prepare files
    files_to_process, skipped = _prepare_files(files, force, vector_store)

    logger.info(f"Processing {len(files_to_process)} files, skipping {skipped} already processed files")

    if not files_to_process:
        logger.info("No new files to process")
        return {
            "total_files": len(files),
            "processed_files": 0,
            "skipped_files": skipped,
            "total_chunks": 0,
        }

    # Process files in batches
    total_chunks = 0
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} with {len(batch)} files")

        # Submit tasks for the batch
        futures = []
        for file_info in batch:
            future = process_file_task.submit(
                file_info=file_info,
                store_name=store_name,
                max_chunk_tokens=max_chunk_tokens,
                root_dir=root_path,
            )
            futures.append(future)

        # Wait for batch to complete and collect results
        for future in futures:
            try:
                chunk_count = future.result()
                total_chunks += chunk_count
            except Exception as e:
                logger.error(f"Error processing file in batch: {e}")

    logger.info(f"Completed RAG file ingestion: {len(files_to_process)} files, {total_chunks} chunks")

    return {
        "total_files": len(files),
        "processed_files": len(files_to_process),
        "skipped_files": skipped,
        "total_chunks": total_chunks,
    }
