"""Markdown chunking utilities for RAG and document analysis.

This module provides functions to chunk markdown content using chonkie's
MarkdownChef for intelligent parsing of markdown structure (text, tables, code blocks).
Small chunks are merged with adjacent ones to avoid fragmentation.
Position tracking (start/end) is preserved for lineage.
Token counting uses tiktoken with the o200k_base encoding (GPT-4o and newer models).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from chonkie import MarkdownChef, RecursiveChunker, TableChunker
from upath import UPath

# Tiktoken encoding for token counting (o200k_base is used by GPT-4o and newer)
TIKTOKEN_ENCODING = "o200k_base"

# Initialize chunkers once (they are reusable and thread-safe)
_markdown_chef: MarkdownChef | None = None
_text_chunker: RecursiveChunker | None = None
_table_chunker: TableChunker | None = None


@lru_cache(maxsize=1)
def _get_tiktoken_encoding() -> tiktoken.Encoding:
    """Get cached tiktoken encoding."""
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_get_tiktoken_encoding().encode(text))


# Default max rows per table chunk (header is always preserved)
DEFAULT_TABLE_MAX_ROWS = 10
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MIN_TOKENS = 100  # Minimum token count to avoid tiny fragments


@dataclass(slots=True)
class ChunkInfo:
    """Information about a single chunk."""

    index: int
    chunk_type: str  # "text", "table", "code", or "mixed" (merged)
    content: str
    token_count: int
    start_pos: int  # Start position in original content
    end_pos: int  # End position in original content


def _get_markdown_chef() -> MarkdownChef:
    """Get or create a MarkdownChef for parsing markdown."""
    global _markdown_chef
    if _markdown_chef is None:
        _markdown_chef = MarkdownChef(tokenizer=TIKTOKEN_ENCODING)
    return _markdown_chef


def _get_text_chunker(max_tokens: int = DEFAULT_MAX_TOKENS) -> RecursiveChunker:
    """Get or create a text chunker."""
    global _text_chunker
    if _text_chunker is None or _text_chunker.chunk_size != max_tokens:
        _text_chunker = RecursiveChunker(
            chunk_size=max_tokens,
            tokenizer=TIKTOKEN_ENCODING,
        )
    return _text_chunker


def _get_table_chunker(max_rows: int = DEFAULT_TABLE_MAX_ROWS) -> TableChunker:
    """Get or create a TableChunker for splitting large tables."""
    global _table_chunker
    if _table_chunker is None or _table_chunker.chunk_size != max_rows:
        _table_chunker = TableChunker(
            tokenizer="row",
            chunk_size=max_rows,
        )
    return _table_chunker


@dataclass
class _RawChunk:
    """Internal raw chunk with position info before merging."""

    chunk_type: str
    content: str
    start_pos: int
    end_pos: int


def _collect_raw_chunks(
    content: str,
    max_tokens: int,
) -> list[_RawChunk]:
    """Parse markdown and collect all chunks in document order with positions.

    Interleaves text and table chunks to preserve the original document order.
    """
    chef = _get_markdown_chef()
    doc = chef.parse(content)

    # Collect all elements with their start positions for sorting
    elements: list[tuple[int, str, str, int, int]] = []  # (start, type, content, start_pos, end_pos)

    # Add text chunks
    for chunk in doc.chunks:
        if chunk.text.strip():
            if chunk.token_count > max_tokens:
                # Split large text chunks
                text_chunker = _get_text_chunker(max_tokens)
                sub_chunks = text_chunker.chunk(chunk.text)
                # Distribute positions proportionally among sub-chunks
                text_start = chunk.start_index
                for sub in sub_chunks:
                    if sub.text.strip():
                        # Calculate relative position within the parent chunk
                        sub_start = text_start + sub.start_index
                        sub_end = text_start + sub.end_index
                        elements.append((sub_start, "text", sub.text, sub_start, sub_end))
            else:
                elements.append((chunk.start_index, "text", chunk.text, chunk.start_index, chunk.end_index))

    # Add tables
    table_chunker = _get_table_chunker()
    for table in doc.tables:
        if table.content.strip():
            lines = table.content.strip().split("\n")
            data_rows = len(lines) - 2 if len(lines) > 2 else 0

            if data_rows > DEFAULT_TABLE_MAX_ROWS:
                table_chunks = table_chunker.chunk(table.content)
                # For split tables, distribute positions proportionally
                table_start = table.start_index
                table_len = table.end_index - table.start_index
                total_chars = sum(len(tc.text) for tc in table_chunks)
                pos_offset = 0
                for tc in table_chunks:
                    if tc.text.strip():
                        # Estimate sub-positions proportionally
                        sub_len = int(table_len * len(tc.text) / total_chars) if total_chars > 0 else 0
                        elements.append(
                            (
                                table_start + pos_offset,
                                "table",
                                tc.text,
                                table_start + pos_offset,
                                table_start + pos_offset + sub_len,
                            )
                        )
                        pos_offset += sub_len
            else:
                elements.append((table.start_index, "table", table.content, table.start_index, table.end_index))

    # Add code blocks
    for code in doc.code:
        if code.content.strip():
            lang_prefix = f"```{code.language}\n" if code.language else "```\n"
            code_content = f"{lang_prefix}{code.content}\n```"
            elements.append((code.start_index, "code", code_content, code.start_index, code.end_index))

    # Sort by start position to preserve document order
    elements.sort(key=lambda x: x[0])

    return [_RawChunk(chunk_type=e[1], content=e[2], start_pos=e[3], end_pos=e[4]) for e in elements]


def _merge_small_chunks(
    raw_chunks: list[_RawChunk],
    min_tokens: int,
    max_tokens: int,
) -> list[ChunkInfo]:
    """Merge small chunks with adjacent ones to avoid tiny fragments.

    Small chunks (any type) are merged with the next chunk.
    If no next chunk or too big, merge with previous.
    Merged chunks get type "mixed" if combining different types.
    """
    if not raw_chunks:
        return []

    result: list[ChunkInfo] = []
    pending: list[_RawChunk] = []  # Accumulated small chunks to merge

    def _merge_pending_with(chunk: _RawChunk) -> _RawChunk:
        """Merge pending small chunks with the given chunk."""
        if not pending:
            return chunk

        # Combine all pending + current
        all_chunks = pending + [chunk]
        merged_content = "\n\n".join(c.content for c in all_chunks)
        start_pos = all_chunks[0].start_pos
        end_pos = all_chunks[-1].end_pos

        # Determine merged type
        types = {c.chunk_type for c in all_chunks}
        if len(types) == 1:
            merged_type = types.pop()
        else:
            merged_type = "mixed"

        pending.clear()
        return _RawChunk(chunk_type=merged_type, content=merged_content, start_pos=start_pos, end_pos=end_pos)

    def _flush_pending_to_result() -> None:
        """Add pending chunks to result (merged together or appended to last)."""
        if not pending:
            return

        merged_content = "\n\n".join(c.content for c in pending)
        merged_tokens = _count_tokens(merged_content)
        start_pos = pending[0].start_pos
        end_pos = pending[-1].end_pos
        types = {c.chunk_type for c in pending}
        merged_type = types.pop() if len(types) == 1 else "mixed"

        if result:
            # Try to append to last result chunk
            last = result[-1]
            combined = last.content + "\n\n" + merged_content
            combined_tokens = _count_tokens(combined)
            if combined_tokens <= max_tokens * 1.2:
                # Merge with last
                new_type = last.chunk_type if last.chunk_type == merged_type else "mixed"
                result[-1] = ChunkInfo(
                    index=last.index,
                    chunk_type=new_type,
                    content=combined,
                    token_count=combined_tokens,
                    start_pos=last.start_pos,
                    end_pos=end_pos,
                )
                pending.clear()
                return

        # Add as new chunk
        result.append(
            ChunkInfo(
                index=len(result),
                chunk_type=merged_type,
                content=merged_content,
                token_count=merged_tokens,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )
        pending.clear()

    for chunk in raw_chunks:
        chunk_tokens = _count_tokens(chunk.content)
        is_small = chunk_tokens < min_tokens

        if is_small:
            pending.append(chunk)
            continue

        # We have a substantial chunk
        # Check if we can merge pending with it
        if pending:
            total_pending_tokens = sum(_count_tokens(c.content) for c in pending)
            if total_pending_tokens + chunk_tokens + 2 <= max_tokens * 1.2:  # +2 tokens for "\n\n"
                chunk = _merge_pending_with(chunk)
                chunk_tokens = _count_tokens(chunk.content)
            else:
                # Pending too big to merge - flush it
                _flush_pending_to_result()

        result.append(
            ChunkInfo(
                index=len(result),
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                token_count=chunk_tokens,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
            )
        )

    # Handle remaining pending
    _flush_pending_to_result()

    # Re-index
    for i, chunk in enumerate(result):
        chunk.index = i

    return result


def chunk_markdown_content(
    content: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> list[ChunkInfo]:
    """Chunk markdown content and return detailed chunk information.

    Uses chonkie's MarkdownChef to parse markdown and extract tables, code blocks,
    and text chunks. Elements are kept in document order. Small chunks are merged
    with adjacent ones (regardless of type). Token counting uses tiktoken.

    Args:
        content: The markdown content to chunk.
        max_tokens: Maximum number of tokens per chunk.
        min_tokens: Minimum tokens for a chunk. Smaller chunks are merged with adjacent ones.

    Returns:
        List of ChunkInfo objects with details about each chunk.
    """
    raw_chunks = _collect_raw_chunks(content, max_tokens)
    return _merge_small_chunks(raw_chunks, min_tokens, max_tokens)


def chunk_markdown_file(
    file_path: UPath | str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> list[ChunkInfo]:
    """Chunk a markdown file and return detailed chunk information.

    Args:
        file_path: Path to the markdown file.
        max_tokens: Maximum number of tokens per chunk.
        min_tokens: Minimum tokens for a chunk. Smaller chunks are merged with adjacent ones.

    Returns:
        List of ChunkInfo objects with details about each chunk.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a markdown file.
    """
    path = UPath(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"Not a markdown file: {path}")

    content = path.read_text(encoding="utf-8")
    return chunk_markdown_content(content, max_tokens=max_tokens, min_tokens=min_tokens)
