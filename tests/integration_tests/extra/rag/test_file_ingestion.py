"""Integration tests for RAG file ingestion with chunker selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from genai_tk.core.retriever_factory import RetrieverFactory
from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.extra.rag.rag_prefect_flow import rag_file_ingestion_flow


def _run_ingestion(**kwargs):
    """Run rag_file_ingestion_flow in ephemeral Prefect context."""
    return run_flow_ephemeral(rag_file_ingestion_flow, **kwargs)


class TestRagFileIngestionFlowChunkers:
    """Integration tests for RAG file ingestion with different chunkers."""

    @pytest.fixture
    def temp_docs_dir(self) -> Path:
        """Create a temporary directory with test documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create markdown file
            md_file = base_path / "test.md"
            md_file.write_text("""# Test Document

This is a test markdown file.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
""")

            # Create text file
            txt_file = base_path / "notes.txt"
            txt_file.write_text("""This is a plain text file.

It has multiple paragraphs.

And more content here.
""")

            # Create Python file
            py_file = base_path / "code.py"
            py_file.write_text("""def hello_world():
    '''A simple function.'''
    return "Hello, world!"

def another_function():
    '''Another function.'''
    pass
""")

            yield base_path

    def test_ingestion_with_auto_chunker(self, temp_docs_dir: Path) -> None:
        """Test file ingestion with auto-detected chunkers."""
        retriever_name = "default"

        result = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="auto",
            include_patterns=["**/*.md", "**/*.txt", "**/*.py"],
            force=True,
        )

        # Verify results
        assert result["total_files"] == 3
        assert result["processed_files"] == 3
        assert result["total_chunks"] > 0
        print(f"✓ Ingested {result['processed_files']} files with {result['total_chunks']} chunks")

    def test_ingestion_with_explicit_markdown_chunker(self, temp_docs_dir: Path) -> None:
        """Test file ingestion with explicit markdown chunker for markdown files only."""
        retriever_name = "default"

        result = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="markdown",
            include_patterns=["**/*.md"],
            force=True,
        )

        # Verify results
        assert result["total_files"] == 1
        assert result["processed_files"] == 1
        assert result["total_chunks"] > 0
        print(f"✓ Ingested markdown file with {result['total_chunks']} chunks")

    def test_ingestion_with_explicit_recursive_chunker(self, temp_docs_dir: Path) -> None:
        """Test file ingestion with explicit recursive chunker."""
        retriever_name = "default"

        result = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="recursive",
            include_patterns=["**/*.txt"],
            force=True,
        )

        # Verify results
        assert result["total_files"] == 1
        assert result["processed_files"] == 1
        assert result["total_chunks"] > 0
        print(f"✓ Ingested text file with {result['total_chunks']} chunks")

    def test_ingestion_documents_have_metadata(self, temp_docs_dir: Path) -> None:
        """Test that ingested documents have proper metadata."""
        retriever_name = "default"

        result = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="auto",
            include_patterns=["**/*.md"],
            force=True,
        )

        assert result["processed_files"] > 0

        # Check that documents in the retriever have metadata
        managed = RetrieverFactory.create(retriever_name)
        vector_store = managed._vector_store

        if vector_store and hasattr(vector_store, "_collection"):
            # Chroma backend: check metadata
            results = vector_store._collection.get(include=["metadatas"])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                assert metadata is not None
                assert "source" in metadata
                assert "file_hash" in metadata
                assert "chunk_index" in metadata
                assert "total_chunks" in metadata
                # Check for new metadata from splitter
                if "token_count" in metadata:
                    print(f"✓ Document has token_count metadata: {metadata['token_count']}")
                if "chunk_type" in metadata:
                    print(f"✓ Document has chunk_type metadata: {metadata['chunk_type']}")

    def test_ingestion_deduplication_with_force_false(self, temp_docs_dir: Path) -> None:
        """Test that force=False skips already processed files."""
        retriever_name = "default"

        # First ingestion
        result1 = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="auto",
            include_patterns=["**/*.md"],
            force=True,
        )

        assert result1["processed_files"] == 1
        chunks1 = result1["total_chunks"]

        # Second ingestion with force=False (should skip)
        result2 = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="auto",
            include_patterns=["**/*.md"],
            force=False,
        )

        assert result2["processed_files"] == 0
        assert result2["skipped_files"] == 1
        assert result2["total_chunks"] == 0
        print(f"✓ Deduplication working: skipped {result2['skipped_files']} file(s)")

    def test_ingestion_with_different_chunk_sizes(self, temp_docs_dir: Path) -> None:
        """Test that different chunk sizes produce different numbers of chunks."""
        retriever_name = "default"

        # Small chunks
        result_small = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=100,  # Small
            chunker_name="auto",
            include_patterns=["**/*.md"],
            force=True,
        )

        # Large chunks (clear cache first by using force=True)
        result_large = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=1000,  # Large
            chunker_name="auto",
            include_patterns=["**/*.md"],
            force=True,
        )

        # Small chunks should produce more chunks
        assert result_small["total_chunks"] >= result_large["total_chunks"]
        print(f"✓ Small chunks ({result_small['total_chunks']}) vs large chunks ({result_large['total_chunks']})")

    def test_ingestion_with_multiple_file_types(self, temp_docs_dir: Path) -> None:
        """Test auto-detection with mixed file types."""
        retriever_name = "default"

        result = _run_ingestion(
            root_dir=str(temp_docs_dir),
            retriever_name=retriever_name,
            max_chunk_tokens=512,
            chunker_name="auto",  # Should auto-detect .md → markdown, .txt/.py → recursive
            include_patterns=["**/*"],
            exclude_patterns=["**/__pycache__"],
            force=True,
        )

        # Should find all 3 files
        assert result["total_files"] == 3
        assert result["processed_files"] == 3
        assert result["total_chunks"] > 0
        print(f"✓ Processed {result['total_files']} files with mixed types: {result['total_chunks']} chunks")

    def test_ingestion_empty_directory(self) -> None:
        """Test ingestion with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_ingestion(
                root_dir=tmpdir,
                retriever_name="default",
                max_chunk_tokens=512,
                chunker_name="auto",
                force=True,
            )

            assert result["total_files"] == 0
            assert result["processed_files"] == 0
            assert result["total_chunks"] == 0
            print("✓ Empty directory handled correctly")

    def test_ingestion_invalid_directory_raises(self) -> None:
        """Test that invalid directory raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            _run_ingestion(
                root_dir="/nonexistent/directory",
                retriever_name="default",
                max_chunk_tokens=512,
                chunker_name="auto",
            )
