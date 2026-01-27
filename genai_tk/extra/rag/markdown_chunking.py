"""Markdown chunking utilities for RAG and document analysis.

This module provides functions to chunk markdown content using chonkie's
MarkdownChef for intelligent parsing of markdown structure (text, tables, code blocks).
Small chunks are merged with adjacent ones to avoid fragmentation.
Position tracking (start/end) is preserved for lineage.
Token counting uses tiktoken with the o200k_base encoding (GPT-4o and newer models).

Context-aware merging: Small chunks are always merged forward with the next chunk
to keep section titles, headers, and introductory text with the content they describe.
This provides better context for embeddings. Token limits are soft limits that can
be exceeded to maintain semantic coherence.

Note: You may see a warning from chonkie about falling back to tiktoken when using
o200k_base encoding. This is expected behavior since o200k_base is tiktoken-specific
and not available in the HuggingFace tokenizers library.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from chonkie import MarkdownChef, RecursiveChunker, TableChunker
from upath import UPath

# Tiktoken encoding for token counting (o200k_base is used by GPT-4o and newer)
TIKTOKEN_ENCODING = "o200k_base"

# Suppress expected chonkie warning about falling back to tiktoken for o200k_base
# This is the intended behavior since o200k_base is tiktoken-specific
warnings.filterwarnings(
    "ignore",
    message="Could not load tokenizer with 'tokenizers'",
    category=UserWarning,
    module="chonkie.tokenizer",
)
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
DEFAULT_MAX_TOKENS = 300
DEFAULT_MIN_TOKENS = 50  # Minimum token count to avoid tiny fragments


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

    content: str
    start_pos: int
    end_pos: int


def _split_trailing_header(text: str) -> tuple[str, str | None]:
    """Split text if it ends with a markdown header section.

    Returns (main_content, trailing_header) where trailing_header is None if no split.
    A trailing header is detected when text ends with a header line (# ...) optionally
    followed by a short paragraph (less than 3 lines of non-header content).
    """
    lines = text.rstrip().split("\n")
    if len(lines) < 2:
        return text, None

    # Find the last header line
    last_header_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            last_header_idx = i
            break

    if last_header_idx < 0:
        return text, None

    # Check if there's substantial content after the header
    non_empty_after = sum(1 for line in lines[last_header_idx + 1 :] if line.strip())

    # If header is near the end (<=3 non-empty lines after), split it off
    if non_empty_after <= 3 and last_header_idx > 0:
        main_content = "\n".join(lines[:last_header_idx]).rstrip()
        trailing_header = "\n".join(lines[last_header_idx:])
        if main_content.strip():
            return main_content, trailing_header

    return text, None


def _collect_raw_chunks(
    content: str,
    max_tokens: int,
) -> list[_RawChunk]:
    """Parse markdown and collect all chunks in document order with positions.

    Interleaves text, table, and code chunks to preserve the original document order.
    Text chunks ending with headers are split so the header can merge forward with tables.
    """
    chef = _get_markdown_chef()
    doc = chef.parse(content)

    # Collect all elements with their start positions for sorting
    elements: list[tuple[int, str, int, int]] = []  # (start, content, start_pos, end_pos)

    # Add text chunks
    for chunk in doc.chunks:
        if chunk.text.strip():
            text = chunk.text
            start_pos = chunk.start_index
            end_pos = chunk.end_index

            # Split off trailing headers so they merge with following content
            main_content, trailing = _split_trailing_header(text)
            if trailing:
                # Calculate approximate split position
                main_len = len(main_content)
                main_end = start_pos + main_len
                if main_content.strip():
                    elements.append((start_pos, main_content, start_pos, main_end))
                elements.append((main_end, trailing, main_end, end_pos))
            elif chunk.token_count > max_tokens:
                # Split large text chunks
                text_chunker = _get_text_chunker(max_tokens)
                sub_chunks = text_chunker.chunk(text)
                text_start = start_pos
                for sub in sub_chunks:
                    if sub.text.strip():
                        sub_start = text_start + sub.start_index
                        sub_end = text_start + sub.end_index
                        elements.append((sub_start, sub.text, sub_start, sub_end))
            else:
                elements.append((start_pos, text, start_pos, end_pos))

    # Add tables
    table_chunker = _get_table_chunker()
    for table in doc.tables:
        if table.content.strip():
            lines = table.content.strip().split("\n")
            data_rows = len(lines) - 2 if len(lines) > 2 else 0

            if data_rows > DEFAULT_TABLE_MAX_ROWS:
                table_chunks = table_chunker.chunk(table.content)
                table_start = table.start_index
                table_len = table.end_index - table.start_index
                total_chars = sum(len(tc.text) for tc in table_chunks)
                pos_offset = 0
                for tc in table_chunks:
                    if tc.text.strip():
                        sub_len = int(table_len * len(tc.text) / total_chars) if total_chars > 0 else 0
                        elements.append(
                            (
                                table_start + pos_offset,
                                tc.text,
                                table_start + pos_offset,
                                table_start + pos_offset + sub_len,
                            )
                        )
                        pos_offset += sub_len
            else:
                elements.append((table.start_index, table.content, table.start_index, table.end_index))

    # Add code blocks
    for code in doc.code:
        if code.content.strip():
            lang_prefix = f"```{code.language}\n" if code.language else "```\n"
            code_content = f"{lang_prefix}{code.content}\n```"
            elements.append((code.start_index, code_content, code.start_index, code.end_index))

    # Sort by start position to preserve document order
    elements.sort(key=lambda x: x[0])

    return [_RawChunk(content=e[1], start_pos=e[2], end_pos=e[3]) for e in elements]


def _merge_small_chunks(
    raw_chunks: list[_RawChunk],
    min_tokens: int,
    max_tokens: int,
) -> list[ChunkInfo]:
    """Merge small chunks forward to keep context with following content.

    Strategy: Small chunks are always accumulated and merged with the next substantial
    chunk. This ensures titles, headers, and introductory paragraphs stay with the
    content they describe (tables, sections, etc.). Token limits are soft - we prefer
    keeping semantic units together over strict size limits.
    """
    if not raw_chunks:
        return []

    result: list[ChunkInfo] = []
    pending: list[_RawChunk] = []  # Small chunks waiting to be merged forward

    def _create_chunk(chunks: list[_RawChunk]) -> ChunkInfo:
        """Create a ChunkInfo from a list of raw chunks."""
        merged_content = "\n\n".join(c.content for c in chunks)
        return ChunkInfo(
            index=len(result),
            chunk_type="text",
            content=merged_content,
            token_count=_count_tokens(merged_content),
            start_pos=chunks[0].start_pos,
            end_pos=chunks[-1].end_pos,
        )

    def _flush_pending() -> None:
        """Flush pending chunks to result."""
        if not pending:
            return
        result.append(_create_chunk(pending))
        pending.clear()

    for chunk in raw_chunks:
        chunk_tokens = _count_tokens(chunk.content)
        is_small = chunk_tokens < min_tokens

        if is_small:
            # Accumulate small chunks to merge forward
            pending.append(chunk)
            continue

        # We have a substantial chunk
        if pending:
            # Calculate combined size
            pending_content = "\n\n".join(c.content for c in pending)
            pending_tokens = _count_tokens(pending_content)
            combined_tokens = pending_tokens + chunk_tokens + 2  # +2 for "\n\n"

            # Merge pending with this chunk (soft limit - allow up to 1.5x max)
            if combined_tokens <= max_tokens * 1.5:
                pending.append(chunk)
                result.append(_create_chunk(pending))
                pending.clear()
            else:
                # Pending too large - flush it separately, then add current chunk
                _flush_pending()
                result.append(_create_chunk([chunk]))
        else:
            result.append(_create_chunk([chunk]))

    # Handle remaining pending chunks
    if pending:
        # Try to merge with last result chunk if it won't exceed soft limit
        if result:
            last = result[-1]
            pending_content = "\n\n".join(c.content for c in pending)
            combined = last.content + "\n\n" + pending_content
            combined_tokens = _count_tokens(combined)

            if combined_tokens <= max_tokens * 1.5:
                result[-1] = ChunkInfo(
                    index=last.index,
                    chunk_type="text",
                    content=combined,
                    token_count=combined_tokens,
                    start_pos=last.start_pos,
                    end_pos=pending[-1].end_pos,
                )
                pending.clear()
            else:
                _flush_pending()
        else:
            _flush_pending()

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
