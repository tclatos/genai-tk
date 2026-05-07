"""Generic LangChain TextSplitter adapter for Chonkie chunkers.

This module provides a bridge between Chonkie chunkers and LangChain's TextSplitter interface,
allowing Chonkie's intelligent document parsing (markdown, tables, code blocks) to be used
in LangChain RAG pipelines.

The ChonkieTextSplitter wraps any Chonkie BaseChunker and produces LangChain Document objects
with rich metadata:
- start_index: Character offset in the original document
- token_count: Token count (uses tiktoken o200k_base encoding by default)
- chunk_type: "text", "table", "code", or "mixed" (for merged chunks)

Forward-merge logic (enabled by default for markdown) merges small chunks with the next
substantial chunk to keep headers and introductory text with their content.
"""

from __future__ import annotations

import warnings
from typing import Any

import tiktoken
from chonkie import BaseChunker, MarkdownChef, RecursiveChunker
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from genai_tk.utils.singleton import once

# Suppress expected chonkie warning about falling back to tiktoken for o200k_base
warnings.filterwarnings(
    "ignore",
    message="Could not load tokenizer with 'tokenizers'",
    category=UserWarning,
    module="chonkie.tokenizer",
)


@once
def _get_tiktoken_encoding(encoding_name: str = "o200k_base") -> tiktoken.Encoding:
    """Get cached tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def _count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = _get_tiktoken_encoding(encoding_name)
    return len(enc.encode(text))


class ChonkieTextSplitter(TextSplitter):
    """LangChain TextSplitter wrapping a Chonkie chunker.

    Produces Document objects with metadata:
    - start_index: Character offset in source
    - token_count: Token count using specified encoding
    - chunk_type: "text", "table", "code", or "mixed"

    Supports forward-merge logic (enabled by default for markdown) to keep
    small chunks with substantial content that follows.

    Example:
        ```python
        # Markdown chunker with forward-merge
        splitter = ChonkieTextSplitter(chunker_type="markdown", max_tokens=300, min_tokens=50, merge_small_chunks=True)
        docs = splitter.split_documents([Document(page_content=content)])

        # Or wrap a custom Chonkie chunker
        from chonkie import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=512, tokenizer="o200k_base")
        splitter = ChonkieTextSplitter(chunker=chunker)
        docs = splitter.split_documents([Document(page_content=content)])
        ```
    """

    def __init__(
        self,
        *,
        chunker: BaseChunker | None = None,
        chunker_type: str | None = None,
        max_tokens: int = 512,
        min_tokens: int = 50,
        merge_small_chunks: bool = False,
        encoding_name: str = "o200k_base",
        **kwargs: Any,
    ):
        """Initialize ChonkieTextSplitter.

        Args:
            chunker: Pre-configured Chonkie BaseChunker instance.
                If provided, other chunker_* params are ignored.
            chunker_type: Type of chunker to create: "markdown" or "recursive".
                Only used if `chunker` is None.
            max_tokens: Max token count per chunk (soft limit for merging).
            min_tokens: Min token count before merging with next chunk.
            merge_small_chunks: If True, merge chunks smaller than min_tokens
                forward with the next substantial chunk. Enabled by default for markdown.
            encoding_name: Tiktoken encoding name (default: "o200k_base" for GPT-4o).
            **kwargs: Additional arguments passed to TextSplitter.

        Raises:
            ValueError: If both chunker and chunker_type are provided, or neither.
        """
        super().__init__(**kwargs)

        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.merge_small_chunks = merge_small_chunks
        self.chunker_type = chunker_type

        if chunker is not None and chunker_type is not None:
            raise ValueError("Cannot specify both 'chunker' and 'chunker_type'")

        tokenizer = _get_tiktoken_encoding(encoding_name)

        if chunker is not None:
            self.chunker = chunker
        elif chunker_type == "markdown":
            self.chunker = MarkdownChef(tokenizer=tokenizer)
            # Markdown uses forward-merge by default
            self.merge_small_chunks = True
        elif chunker_type == "recursive":
            self.chunker = RecursiveChunker(
                chunk_size=max_tokens,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(
                f"chunker_type must be 'markdown' or 'recursive', got {chunker_type!r}. "
                "Alternatively, provide a pre-configured 'chunker' instance."
            )

    def split_text(self, text: str) -> list[str]:
        """Split text and return list of chunk strings (no metadata).

        Args:
            text: Input text to split.

        Returns:
            List of chunk text strings.
        """
        chunks = self._chunk_text(text)
        return [chunk.text if hasattr(chunk, "text") else chunk.content for chunk in chunks]

    def _chunk_text(self, text: str) -> list[Any]:
        """Internal method to chunk text using the configured chunker.

        Handles both MarkdownChef (parse) and other chunkers (chunk).
        Returns raw chunk objects from the chunker.
        """
        if isinstance(self.chunker, MarkdownChef):
            # MarkdownChef uses parse() to return a document
            doc = self.chunker.parse(text)
            # Flatten all chunks from the parsed document
            chunks = []
            if hasattr(doc, "chunks"):
                chunks.extend(doc.chunks)
            if hasattr(doc, "tables"):
                chunks.extend(doc.tables)
            if hasattr(doc, "code"):
                chunks.extend(doc.code)
            return chunks
        else:
            # Other chunkers use chunk()
            return self.chunker.chunk(text)

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Create Document objects with rich metadata from chunks.

        Args:
            texts: List of input texts to split.
            metadatas: List of base metadata dicts (one per text).
                Will be merged with chunk-specific metadata.
            **kwargs: Additional arguments (ignored; for API compatibility).

        Returns:
            List of Document objects with metadata:
            - start_index: Character offset in source
            - token_count: Token count of chunk
            - chunk_type: "text", "table", "code", or "mixed"
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for text, metadata in zip(texts, _metadatas, strict=False):
            chunks = self._chunk_text(text)

            if self.merge_small_chunks and chunks:
                chunks = self._merge_small_chunks(chunks)

            for _i, chunk in enumerate(chunks):
                # Extract text content (different chunkers store it differently)
                chunk_text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
                if not chunk_text:
                    continue

                # Extract metadata from chunk
                chunk_metadata = {
                    **metadata,
                    "start_index": getattr(chunk, "start_index", 0),
                    "token_count": _count_tokens(chunk_text, self.encoding_name),
                    "chunk_type": self._infer_chunk_type(chunk),
                }

                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
                documents.append(doc)

        return documents

    def _merge_small_chunks(self, chunks: list[Any]) -> list[Any]:
        """Merge chunks smaller than min_tokens with subsequent chunks.

        Keeps section headers and introductory text with the content they describe.
        This is a no-op for very small lists.

        Args:
            chunks: List of Chonkie chunk objects.

        Returns:
            List of merged chunk objects.
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[Any] = []
        small_chunk_buffer = None
        small_chunk_buffer_text = None

        for chunk in chunks:
            # Safely extract text from chunk (different chunk types may have different attributes)
            chunk_text = getattr(chunk, "text", None) or getattr(chunk, "content", None)

            # Skip chunks that don't have text content
            if not chunk_text:
                merged.append(chunk)
                continue

            token_count = _count_tokens(chunk_text, self.encoding_name)

            if token_count < self.min_tokens:
                # Accumulate small chunks
                if small_chunk_buffer is None:
                    small_chunk_buffer = chunk
                    small_chunk_buffer_text = chunk_text
                else:
                    # Concatenate small chunks
                    small_chunk_buffer_text += "\n\n" + chunk_text
                continue

            # We have a substantial chunk
            if small_chunk_buffer is not None:
                # Prepend accumulated small chunks to this chunk
                chunk_text = small_chunk_buffer_text + "\n\n" + chunk_text
                # Update the text attribute if it exists
                if hasattr(chunk, "text"):
                    chunk.text = chunk_text
                elif hasattr(chunk, "content"):
                    chunk.content = chunk_text
                small_chunk_buffer = None
                small_chunk_buffer_text = None

            merged.append(chunk)

        # Don't lose trailing small chunks (merge into last text chunk, or emit as-is)
        if small_chunk_buffer is not None:
            flushed = False
            if merged:
                # Try to find the last chunk with text content
                for last_chunk in reversed(merged):
                    last_chunk_text = getattr(last_chunk, "text", None) or getattr(last_chunk, "content", None)
                    if last_chunk_text:
                        merged_text = last_chunk_text + "\n\n" + small_chunk_buffer_text
                        if hasattr(last_chunk, "text"):
                            last_chunk.text = merged_text
                        elif hasattr(last_chunk, "content"):
                            last_chunk.content = merged_text
                        flushed = True
                        break
            if not flushed:
                # No text chunk to merge into — emit the buffer as its own chunk
                merged.append(small_chunk_buffer)

        return merged

    def _infer_chunk_type(self, chunk: Any) -> str:
        """Infer chunk type from Chonkie chunk object.

        Args:
            chunk: A Chonkie chunk object.

        Returns:
            Chunk type: "text", "table", "code", or "mixed".
        """
        # Chonkie chunks have a `chunk_type` attribute (for MarkdownChef output)
        chunk_type = getattr(chunk, "chunk_type", None)
        if chunk_type:
            return chunk_type

        # Fallback: inspect content for markers
        text = getattr(chunk, "text", None) or getattr(chunk, "content", None) or ""
        if "```" in text or "    " in text.split("\n")[0]:  # Code block indicators
            return "code"
        elif "|" in text and "\n" in text:  # Table-like structure
            return "table"
        return "text"
