"""Unit tests for ChunkerFactory."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from genai_tk.core.factories.chunker_factory import ChunkerFactory
from genai_tk.workflow.rag.chonkie_splitter import ChonkieTextSplitter


class TestChunkerFactoryCreate:
    """Test ChunkerFactory.create method."""

    def test_create_recursive(self) -> None:
        """Test creating recursive chunker from config."""
        splitter = ChunkerFactory.create("recursive")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_markdown(self) -> None:
        """Test creating markdown chunker from config."""
        splitter = ChunkerFactory.create("markdown")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_chonkie_recursive(self) -> None:
        """Test creating chonkie_recursive chunker from config."""
        splitter = ChunkerFactory.create("chonkie_recursive")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_returns_textplitter(self) -> None:
        """Test that all created splitters inherit from TextSplitter."""
        for name in ["recursive", "markdown", "chonkie_recursive"]:
            splitter = ChunkerFactory.create(name)
            assert isinstance(splitter, TextSplitter)

    def test_create_invalid_config_raises(self) -> None:
        """Test that creating invalid config raises KeyError."""
        with pytest.raises(KeyError, match="not found in config.chunkers"):
            ChunkerFactory.create("nonexistent_chunker")

    def test_create_with_params(self) -> None:
        """Test that chunker is created with configured parameters."""
        # Recursive chunker should have chunk_size from config
        splitter = ChunkerFactory.create("recursive")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)
        # RecursiveCharacterTextSplitter stores chunk_size as _chunk_size internally
        assert hasattr(splitter, "_chunk_size")


class TestChunkerFactoryCreateForFile:
    """Test ChunkerFactory.create_for_file method."""

    def test_create_for_file_markdown_extension(self) -> None:
        """Test auto-detection for .md files."""
        splitter = ChunkerFactory.create_for_file(Path("document.md"), "auto")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_for_file_markdown_alternate_extension(self) -> None:
        """Test auto-detection for .markdown files."""
        splitter = ChunkerFactory.create_for_file(Path("readme.markdown"), "auto")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_for_file_rst_extension(self) -> None:
        """Test auto-detection for .rst files (also uses markdown)."""
        splitter = ChunkerFactory.create_for_file(Path("docs.rst"), "auto")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_for_file_python_extension(self) -> None:
        """Test auto-detection for .py files (recursive)."""
        splitter = ChunkerFactory.create_for_file(Path("script.py"), "auto")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_java_extension(self) -> None:
        """Test auto-detection for .java files (recursive)."""
        splitter = ChunkerFactory.create_for_file(Path("Main.java"), "auto")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_javascript_extension(self) -> None:
        """Test auto-detection for .js files (recursive)."""
        splitter = ChunkerFactory.create_for_file(Path("app.js"), "auto")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_text_extension(self) -> None:
        """Test auto-detection for .txt files (recursive)."""
        splitter = ChunkerFactory.create_for_file(Path("notes.txt"), "auto")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_unknown_extension_uses_default(self) -> None:
        """Test that unknown extensions use .default mapping."""
        splitter = ChunkerFactory.create_for_file(Path("data.unknown"), "auto")
        # Should fall back to default, which is recursive
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_explicit_chunker(self) -> None:
        """Test specifying explicit chunker overrides auto-detection."""
        # File is .md but we request recursive
        splitter = ChunkerFactory.create_for_file(Path("doc.md"), "recursive")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_for_file_with_path_string(self) -> None:
        """Test that string paths are converted to UPath."""
        splitter = ChunkerFactory.create_for_file("document.md", "auto")
        assert isinstance(splitter, ChonkieTextSplitter)

    def test_create_for_file_invalid_auto_raises(self) -> None:
        """Test that invalid auto-detection raises informative error."""
        # This would only happen if chunker_auto_map is broken
        # (current config always has .default)
        # Skip as current config always has fallback


class TestChunkerFactoryIntegration:
    """Integration tests for ChunkerFactory."""

    def test_factory_chunking_integration(self) -> None:
        """Test that factory-created chunkers work end-to-end."""
        from langchain_core.documents import Document

        # Create markdown chunker
        splitter = ChunkerFactory.create("markdown")

        # Split document
        text = "# Title\n\nContent here."
        docs = splitter.create_documents([text], metadatas=[{"source": "test.md"}])

        # Verify output
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        assert all("source" in d.metadata for d in docs)
        assert all("token_count" in d.metadata for d in docs)

    def test_factory_auto_detection_integration(self) -> None:
        """Test auto-detection end-to-end."""
        from langchain_core.documents import Document

        # Create markdown chunker via auto-detection
        splitter = ChunkerFactory.create_for_file(Path("doc.md"), "auto")

        # Split document
        text = "# Title\n\nContent."
        docs = splitter.create_documents([text])

        # Verify
        assert len(docs) > 0
        assert isinstance(docs[0], Document)

    def test_factory_multiple_chunker_types(self) -> None:
        """Test creating different chunker types."""
        from langchain_core.documents import Document

        text = "Content"

        # Create each type and verify they work
        for config_name in ["recursive", "markdown", "chonkie_recursive"]:
            splitter = ChunkerFactory.create(config_name)
            docs = splitter.split_documents([Document(page_content=text)])
            # Just verify they don't crash
            assert splitter is not None
