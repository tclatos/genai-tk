"""Unit tests for ChonkieTextSplitter."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from genai_tk.extra.rag.chonkie_splitter import ChonkieTextSplitter


class TestChonkieTextSplitterInitialization:
    """Test ChonkieTextSplitter initialization."""

    def test_init_with_chunker_type_markdown(self) -> None:
        """Test initialization with chunker_type='markdown'."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        assert splitter is not None
        assert splitter.chunker_type == "markdown"
        assert splitter.merge_small_chunks is True  # Should be enabled for markdown

    def test_init_with_chunker_type_recursive(self) -> None:
        """Test initialization with chunker_type='recursive'."""
        splitter = ChonkieTextSplitter(chunker_type="recursive", max_tokens=256)
        assert splitter is not None
        assert splitter.max_tokens == 256

    def test_init_raises_on_missing_chunker(self) -> None:
        """Test that initialization fails without chunker or chunker_type."""
        with pytest.raises(ValueError, match="chunker_type must be"):
            ChonkieTextSplitter()

    def test_init_raises_on_both_chunker_and_type(self) -> None:
        """Test that providing both chunker and chunker_type raises error."""
        from chonkie import MarkdownChef

        chef = MarkdownChef()
        with pytest.raises(ValueError, match="Cannot specify both"):
            ChonkieTextSplitter(chunker=chef, chunker_type="markdown")

    def test_init_with_custom_encoding(self) -> None:
        """Test initialization with custom encoding name."""
        splitter = ChonkieTextSplitter(chunker_type="markdown", encoding_name="cl100k_base")
        assert splitter.encoding_name == "cl100k_base"


class TestChonkieTextSplitterSplitText:
    """Test split_text method."""

    def test_split_text_markdown(self) -> None:
        """Test split_text with markdown content."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "# Title\n\nSome content."
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_text_empty(self) -> None:
        """Test split_text with empty content."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        chunks = splitter.split_text("")
        assert isinstance(chunks, list)

    def test_split_text_recursive(self) -> None:
        """Test split_text with recursive chunker."""
        splitter = ChonkieTextSplitter(chunker_type="recursive", max_tokens=100)
        text = "First paragraph with some content.\n\nSecond paragraph with more content."
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestChonkieTextSplitterCreateDocuments:
    """Test create_documents method."""

    def test_create_documents_basic(self) -> None:
        """Test create_documents returns Document objects."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "# Title\n\nContent here."
        docs = splitter.create_documents([text])

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)

    def test_create_documents_has_metadata(self) -> None:
        """Test that created documents have required metadata."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "# Title\n\nContent here."
        docs = splitter.create_documents([text])

        doc = docs[0]
        assert "start_index" in doc.metadata
        assert "token_count" in doc.metadata
        assert "chunk_type" in doc.metadata
        assert isinstance(doc.metadata["token_count"], int)
        assert doc.metadata["token_count"] > 0

    def test_create_documents_with_base_metadata(self) -> None:
        """Test create_documents merges base metadata."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "# Title\n\nContent here."
        base_metadata = {"source": "test.md", "file_hash": "abc123"}

        docs = splitter.create_documents([text], metadatas=[base_metadata])

        doc = docs[0]
        assert doc.metadata["source"] == "test.md"
        assert doc.metadata["file_hash"] == "abc123"
        assert "token_count" in doc.metadata
        assert "chunk_type" in doc.metadata

    def test_create_documents_chunk_type_text(self) -> None:
        """Test that chunk_type is correctly identified as text."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "# Title\n\nThis is regular paragraph text without special formatting."
        docs = splitter.create_documents([text])

        doc = docs[0]
        assert doc.metadata["chunk_type"] in ["text", "mixed"]

    def test_create_documents_multiple_texts(self) -> None:
        """Test create_documents with multiple input texts."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        texts = [
            "# Title 1\n\nContent 1.",
            "# Title 2\n\nContent 2.",
        ]
        metadatas = [
            {"source": "file1.md"},
            {"source": "file2.md"},
        ]

        docs = splitter.create_documents(texts, metadatas=metadatas)

        assert len(docs) >= 2
        # Check that sources are preserved
        sources = [doc.metadata.get("source") for doc in docs]
        assert "file1.md" in sources
        assert "file2.md" in sources

    def test_create_documents_token_count_reasonable(self) -> None:
        """Test that token_count is reasonable for content."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = "Hello world"
        docs = splitter.create_documents([text])

        doc = docs[0]
        # "Hello world" should be 2-3 tokens
        assert doc.metadata["token_count"] >= 2
        assert doc.metadata["token_count"] <= 5


class TestChonkieTextSplitterMerging:
    """Test small chunk merging logic."""

    def test_merge_small_chunks_enabled_markdown(self) -> None:
        """Test that merging is enabled by default for markdown."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        assert splitter.merge_small_chunks is True

    def test_merge_small_chunks_disabled_recursive(self) -> None:
        """Test that merging can be controlled for other chunkers."""
        splitter = ChonkieTextSplitter(chunker_type="recursive", merge_small_chunks=False)
        assert splitter.merge_small_chunks is False

    def test_merge_preserves_content(self) -> None:
        """Test that merging doesn't lose content."""
        splitter = ChonkieTextSplitter(chunker_type="markdown", max_tokens=500, min_tokens=10)
        text = "# Title\n\nSmall\n\nMedium content that is longer\n\nAnother part"
        docs = splitter.create_documents([text])

        # Reconstruct content
        reconstructed = "\n\n".join(doc.page_content for doc in docs)
        assert "Title" in reconstructed


class TestChonkieTextSplitterInferChunkType:
    """Test chunk type inference."""

    def test_infer_chunk_type_code(self) -> None:
        """Test that code blocks are identified."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")

        # Code blocks use backticks
        chunk_type = splitter._infer_chunk_type(type("Chunk", (), {"text": "```python\ncode\n```"})())
        assert chunk_type == "code"

    def test_infer_chunk_type_text(self) -> None:
        """Test that regular text is identified as text."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")

        chunk_type = splitter._infer_chunk_type(type("Chunk", (), {"text": "Regular text content"})())
        assert chunk_type == "text"

    def test_infer_chunk_type_preserves_chonkie_type(self) -> None:
        """Test that chunk_type from chunker is preserved."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")

        # Mock a chunk with chunk_type attribute
        chunk = type("Chunk", (), {"text": "Content", "chunk_type": "table"})()
        chunk_type = splitter._infer_chunk_type(chunk)
        assert chunk_type == "table"


class TestChonkieTextSplitterIntegration:
    """Integration tests."""

    def test_markdown_with_tables_and_code(self) -> None:
        """Test markdown parsing with multiple content types."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        text = """# Main Title

This is an introduction.

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

```python
def hello():
    return "world"
```

Some final text.
"""
        docs = splitter.create_documents([text])

        assert len(docs) > 0
        # All docs should have required metadata
        for doc in docs:
            assert "token_count" in doc.metadata
            assert "chunk_type" in doc.metadata
            assert "start_index" in doc.metadata

    def test_split_documents_compatibility(self) -> None:
        """Test that split_documents (LangChain API) works."""
        splitter = ChonkieTextSplitter(chunker_type="markdown")
        doc = Document(page_content="# Title\n\nContent here.")

        split_docs = splitter.split_documents([doc])

        assert isinstance(split_docs, list)
        assert len(split_docs) > 0
        assert all(isinstance(d, Document) for d in split_docs)
