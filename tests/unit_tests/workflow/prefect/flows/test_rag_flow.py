"""Unit tests for the RAG file-ingestion Prefect flow.

Covers the pure helpers and the flow's early-return / error paths.  The
``process_file_task`` and full ingestion loop are not exercised here because the
pytest profile's configured retrievers resolve to **real** embeddings models
(network APIs); constructing a fake-embeddings retriever would require mutating
shared config, which is out of scope for test-only changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.core.factories.retriever_factory import ManagedRetriever, _EmptyRetriever
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.prefect.flows.rag_flow import (
    FileToProcess,
    _load_file_content,
    _prepare_files,
    rag_file_ingestion_flow,
)


def _no_vector_store_retriever() -> ManagedRetriever:
    """A real ManagedRetriever with no vector store (skips Chroma hash dedup)."""
    return ManagedRetriever(retriever=_EmptyRetriever(), vector_store=None)


# ---------------------------------------------------------------------------
# _load_file_content
# ---------------------------------------------------------------------------


def test_load_file_content_reads_text(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("hello world", encoding="utf-8")
    assert _load_file_content(f) == "hello world"


def test_load_file_content_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        _load_file_content(tmp_path / "missing.md")


# ---------------------------------------------------------------------------
# _prepare_files
# ---------------------------------------------------------------------------


def test_prepare_files_new_file_queued(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("content", encoding="utf-8")
    managed = _no_vector_store_retriever()

    to_process, skipped = _prepare_files([f], force=False, managed=managed)
    assert len(to_process) == 1
    assert skipped == 0
    assert to_process[0].path == f
    assert to_process[0].content == "content"


def test_prepare_files_manifest_cache_skips_fresh(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("content", encoding="utf-8")
    managed = _no_vector_store_retriever()
    from genai_tk.utils.hashing import file_digest

    cache = ManifestCache()
    cache.record_success(key=str(f), fingerprint=file_digest(f), outputs={})

    to_process, skipped = _prepare_files([f], force=False, managed=managed, cache=cache)
    assert to_process == []
    assert skipped == 1


def test_prepare_files_force_reprocesses_fresh(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("content", encoding="utf-8")
    managed = _no_vector_store_retriever()
    from genai_tk.utils.hashing import file_digest

    cache = ManifestCache()
    cache.record_success(key=str(f), fingerprint=file_digest(f), outputs={})

    to_process, skipped = _prepare_files([f], force=True, managed=managed, cache=cache)
    assert len(to_process) == 1
    assert skipped == 0


def test_prepare_files_skips_unreadable(tmp_path: Path) -> None:
    managed = _no_vector_store_retriever()
    to_process, skipped = _prepare_files([tmp_path / "missing.md"], force=False, managed=managed)
    assert to_process == []
    assert skipped == 0


def test_prepare_files_returns_file_to_process_dataclass(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")
    managed = _no_vector_store_retriever()
    to_process, _ = _prepare_files([f], force=False, managed=managed)
    assert isinstance(to_process[0], FileToProcess)
    assert to_process[0].content_hash  # non-empty


# ---------------------------------------------------------------------------
# rag_file_ingestion_flow — early-return / error paths
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_rag_flow_nonexistent_base_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base_dir does not exist"):
        rag_file_ingestion_flow(
            base_dir=str(tmp_path / "missing"),
            retriever_name="default",
            max_chunk_tokens=100,
        )


@pytest.mark.fake_models
def test_rag_flow_no_files_returns_zero_stats(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    # default pathspec is **/* but the dir is empty
    result = rag_file_ingestion_flow(
        base_dir=str(src),
        retriever_name="default",
        max_chunk_tokens=100,
    )
    assert result == {"total_files": 0, "processed_files": 0, "skipped_files": 0, "total_chunks": 0}


@pytest.mark.fake_models
def test_rag_flow_exclude_patterns_filter_all_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("content", encoding="utf-8")
    # exclude everything
    result = rag_file_ingestion_flow(
        base_dir=str(src),
        retriever_name="default",
        max_chunk_tokens=100,
        pathspecs=["**/*"],
        exclude_patterns=["**/*"],
    )
    assert result["total_files"] == 0


# ---------------------------------------------------------------------------
# process_file_task & error / downtime paths
# ---------------------------------------------------------------------------


def test_prepare_files_chroma_exception_gracefully_handled(tmp_path: Path) -> None:
    """When vector store collection.get() raises an error (e.g. DB downtime), prepare_files still succeeds."""
    from unittest.mock import MagicMock

    f = tmp_path / "doc.md"
    f.write_text("content", encoding="utf-8")

    mock_collection = MagicMock()
    mock_collection.get.side_effect = ConnectionError("Chroma server unreachable")

    mock_vs = MagicMock()
    mock_vs._collection = mock_collection

    managed = ManagedRetriever(retriever=_EmptyRetriever(), vector_store=mock_vs)

    to_process, skipped = _prepare_files([f], force=False, managed=managed)
    assert len(to_process) == 1
    assert skipped == 0
    assert to_process[0].path == f


def test_process_file_task_unknown_chunker_raises(tmp_path: Path) -> None:
    """When chunker configuration is invalid, process_file_task raises KeyError."""
    from genai_tk.workflow.prefect.flows.rag_flow import process_file_task

    f = tmp_path / "test.custom_ext"
    f.write_text("some content", encoding="utf-8")
    info = FileToProcess(path=f, content_hash="hash123", content="some content")

    with pytest.raises(KeyError):
        process_file_task.fn(
            file_info=info,
            retriever_name="default",
            max_chunk_tokens=100,
            chunker_name="non_existent_chunker_xyz",
        )


def test_process_file_task_empty_document_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When chunker returns empty document list, returns 0 chunks without error."""
    from unittest.mock import MagicMock

    from genai_tk.core.factories.chunker_factory import ChunkerFactory
    from genai_tk.workflow.prefect.flows.rag_flow import process_file_task

    mock_splitter = MagicMock()
    mock_splitter.create_documents.return_value = []
    monkeypatch.setattr(ChunkerFactory, "create_for_file", lambda *a, **kw: mock_splitter)

    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    info = FileToProcess(path=f, content_hash="h1", content="")

    count = process_file_task.fn(
        file_info=info,
        retriever_name="default",
        max_chunk_tokens=100,
        chunker_name="auto",
    )
    assert count == 0


def test_process_file_task_vector_store_downtime_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When retriever/vector store fails (e.g. timeout / downtime), process_file_task propagates the error."""
    from unittest.mock import MagicMock

    from genai_tk.core.factories.retriever_factory import RetrieverFactory
    from genai_tk.workflow.prefect.flows.rag_flow import process_file_task

    mock_managed = MagicMock()
    mock_managed.add_documents.side_effect = TimeoutError("Vector store request timed out")
    monkeypatch.setattr(RetrieverFactory, "create", lambda *a, **kw: mock_managed)

    f = tmp_path / "doc.md"
    f.write_text("important information", encoding="utf-8")
    info = FileToProcess(path=f, content_hash="h2", content="important information")

    with pytest.raises(TimeoutError, match="Vector store request timed out"):
        process_file_task.fn(
            file_info=info,
            retriever_name="default",
            max_chunk_tokens=100,
            chunker_name="auto",
        )
