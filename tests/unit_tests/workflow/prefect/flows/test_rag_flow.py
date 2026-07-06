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
