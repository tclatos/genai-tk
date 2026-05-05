from __future__ import annotations

from langchain_text_splitters import TextSplitter

"""
Markdown chunking utilities for RAG using LangChain TextSplitter interface.

This module provides factory functions to create LangChain-compatible TextSplitter
instances for chunking markdown content using chonkie's MarkdownChef for intelligent
parsing of markdown structure (text, tables, code blocks).

Use with ChunkerFactory and YAML configuration for flexible, production-ready
document chunking with metadata tracking (start_index, token_count, chunk_type).

Note: Warnings about chonkie falling back to tiktoken for o200k_base encoding are
expected and normal - o200k_base is tiktoken-specific (used by GPT-4o and newer).
"""


def create_markdown_splitter(
    max_tokens: int = 300,
    min_tokens: int = 50,
    encoding_name: str = "o200k_base",
) -> TextSplitter:
    """Create a LangChain-compatible TextSplitter for markdown documents.

    Returns a ChonkieTextSplitter configured to parse markdown with intelligent
    chunk merging. Use this with ChunkerFactory for YAML-based configuration.

    Args:
        max_tokens: Maximum tokens per chunk (soft limit for merging).
        min_tokens: Minimum tokens before merging with next chunk.
        encoding_name: Tiktoken encoding name (default: o200k_base for GPT-4o).

    Returns:
        ChonkieTextSplitter configured for markdown parsing.

    Example:
        ```python
        splitter = create_markdown_splitter(max_tokens=300, min_tokens=50)
        docs = splitter.split_documents([Document(page_content=markdown_text)])
        # docs have metadata: start_index, token_count, chunk_type
        ```
    """
    from genai_tk.extra.rag.chonkie_splitter import ChonkieTextSplitter

    return ChonkieTextSplitter(
        chunker_type="markdown",
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        merge_small_chunks=True,
        encoding_name=encoding_name,
    )
