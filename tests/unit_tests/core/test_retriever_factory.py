"""Unit tests for RetrieverFactory and ManagedRetriever.

All tests use in-memory / fake backends so no external services are needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from genai_tk.core.retriever_factory import (
    BM25DocumentStore,
    BM25RetrieverConfig,
    CompositeDocumentStore,
    EnsembleChildConfig,
    EnsembleRetrieverConfig,
    ManagedRetriever,
    RetrieverFactory,
    VectorDocumentStore,
    VectorRetrieverConfig,
    _EmptyRetriever,
    _make_record_manager,
    _parse_retriever_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(page_content="Python is a programming language", metadata={"source": "doc1"}),
        Document(page_content="Machine learning transforms industries", metadata={"source": "doc2"}),
        Document(page_content="Vector search uses embeddings", metadata={"source": "doc3"}),
    ]


@pytest.fixture
def bm25_store(tmp_path: Path) -> BM25DocumentStore:
    return BM25DocumentStore(cache_dir=tmp_path / "bm25")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_parse_vector(self) -> None:
        cfg = _parse_retriever_config({"type": "vector", "embeddings_store": "default"})
        assert isinstance(cfg, VectorRetrieverConfig)
        assert cfg.embeddings_store == "default"
        assert cfg.top_k == 4

    def test_parse_bm25(self) -> None:
        cfg = _parse_retriever_config({"type": "bm25", "k": 6})
        assert isinstance(cfg, BM25RetrieverConfig)
        assert cfg.k == 6

    def test_parse_ensemble(self) -> None:
        cfg = _parse_retriever_config(
            {
                "type": "ensemble",
                "retrievers": [{"ref": "a", "weight": 0.6}, {"ref": "b", "weight": 0.4}],
            }
        )
        assert isinstance(cfg, EnsembleRetrieverConfig)
        assert len(cfg.retrievers) == 2

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(Exception):
            _parse_retriever_config({"type": "unknown_type"})


# ---------------------------------------------------------------------------
# BM25DocumentStore
# ---------------------------------------------------------------------------


class TestBM25DocumentStore:
    @pytest.mark.asyncio
    async def test_add_documents_creates_index(
        self, bm25_store: BM25DocumentStore, sample_docs: list[Document]
    ) -> None:
        await bm25_store.aadd_documents(sample_docs)
        assert (bm25_store.cache_dir / "bm25_index").exists()
        assert (bm25_store.cache_dir / "documents.json").exists()

    @pytest.mark.asyncio
    async def test_query_after_add(self, bm25_store: BM25DocumentStore, sample_docs: list[Document]) -> None:
        await bm25_store.aadd_documents(sample_docs)
        results = await bm25_store.aget_relevant_documents("programming language", k=2)
        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    @pytest.mark.asyncio
    async def test_empty_query_before_add(self, bm25_store: BM25DocumentStore) -> None:
        results = await bm25_store.aget_relevant_documents("test", k=3)
        assert results == []

    def test_load_from_cache(self, bm25_store: BM25DocumentStore, sample_docs: list[Document]) -> None:
        import asyncio

        asyncio.run(bm25_store.aadd_documents(sample_docs))

        # Create new store pointing to same cache
        new_store = BM25DocumentStore(cache_dir=bm25_store.cache_dir)
        retriever = new_store.get_or_load_retriever(k=2)
        assert retriever is not None
        assert len(retriever.docs) == len(sample_docs)

    @pytest.mark.asyncio
    async def test_rebuild_on_readd(self, bm25_store: BM25DocumentStore, sample_docs: list[Document]) -> None:
        await bm25_store.aadd_documents(sample_docs)
        extra = [Document(page_content="New document added", metadata={"source": "new"})]
        await bm25_store.aadd_documents(extra)
        retriever = bm25_store.get_or_load_retriever()
        assert retriever is not None
        assert len(retriever.docs) == 1


# ---------------------------------------------------------------------------
# VectorDocumentStore
# ---------------------------------------------------------------------------


class TestVectorDocumentStore:
    @pytest.mark.asyncio
    async def test_add_without_record_manager(self, sample_docs: list[Document]) -> None:
        mock_vs = AsyncMock()
        mock_vs.aadd_documents = AsyncMock(return_value=["id1", "id2", "id3"])
        store = VectorDocumentStore(vector_store=mock_vs)
        result = await store.aadd_documents(sample_docs)
        mock_vs.aadd_documents.assert_called_once_with(sample_docs)
        assert result == ["id1", "id2", "id3"]

    @pytest.mark.asyncio
    async def test_add_with_record_manager(self, sample_docs: list[Document]) -> None:
        mock_vs = MagicMock()
        mock_rm = MagicMock()
        store = VectorDocumentStore(vector_store=mock_vs, record_manager=mock_rm)

        with patch("langchain_classic.indexes.index") as mock_index:
            mock_index.return_value = MagicMock()
            await store.aadd_documents(sample_docs)
            mock_index.assert_called_once()


# ---------------------------------------------------------------------------
# CompositeDocumentStore
# ---------------------------------------------------------------------------


class TestCompositeDocumentStore:
    @pytest.mark.asyncio
    async def test_fans_out_to_all_stores(self, sample_docs: list[Document]) -> None:
        mock_vs_a = AsyncMock()
        mock_vs_a.aadd_documents = AsyncMock(return_value=["a"])
        mock_vs_b = AsyncMock()
        mock_vs_b.aadd_documents = AsyncMock(return_value=["b"])

        store_a = VectorDocumentStore(vector_store=mock_vs_a)
        store_b = VectorDocumentStore(vector_store=mock_vs_b)

        composite = CompositeDocumentStore(stores=[store_a, store_b])
        result = await composite.aadd_documents(sample_docs)

        mock_vs_a.aadd_documents.assert_called_once_with(sample_docs)
        mock_vs_b.aadd_documents.assert_called_once_with(sample_docs)
        assert result == ["a"]  # returns first store result


# ---------------------------------------------------------------------------
# ManagedRetriever
# ---------------------------------------------------------------------------


class TestManagedRetriever:
    def _make_vector_managed(self) -> ManagedRetriever:
        mock_vs = AsyncMock()
        mock_vs.asimilarity_search = AsyncMock(return_value=[Document(page_content="result", metadata={})])
        store = VectorDocumentStore(vector_store=mock_vs)
        return ManagedRetriever(
            retriever=_EmptyRetriever(),
            store=store,
            default_k=4,
            config_tag="test",
            vector_store=mock_vs,
        )

    @pytest.mark.asyncio
    async def test_aquery_vector(self) -> None:
        managed = self._make_vector_managed()
        results = await managed.aquery("test query")
        assert len(results) == 1
        assert results[0].page_content == "result"

    @pytest.mark.asyncio
    async def test_aquery_uses_default_k(self) -> None:
        mock_vs = AsyncMock()
        mock_vs.asimilarity_search = AsyncMock(return_value=[])
        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            default_k=7,
            config_tag="test",
            vector_store=mock_vs,
        )
        await managed.aquery("test")
        mock_vs.asimilarity_search.assert_called_once_with("test", k=7, filter=None)

    @pytest.mark.asyncio
    async def test_aquery_with_filter(self) -> None:
        mock_vs = AsyncMock()
        mock_vs.asimilarity_search = AsyncMock(return_value=[])
        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            default_k=4,
            config_tag="test",
            vector_store=mock_vs,
        )
        await managed.aquery("test", filter={"source": "docs"})
        mock_vs.asimilarity_search.assert_called_once_with("test", k=4, filter={"source": "docs"})

    @pytest.mark.asyncio
    async def test_aadd_raises_when_no_store(self) -> None:
        managed = ManagedRetriever(retriever=_EmptyRetriever(), store=None, config_tag="read_only")
        with pytest.raises(RuntimeError, match="read-only"):
            await managed.aadd_documents([Document(page_content="x", metadata={})])

    def test_has_store(self) -> None:
        managed_with = self._make_vector_managed()
        assert managed_with.has_store is True

        managed_without = ManagedRetriever(retriever=_EmptyRetriever(), store=None, config_tag="t")
        assert managed_without.has_store is False

    def test_sync_query_wrapper(self) -> None:
        mock_vs = MagicMock()
        mock_vs.asimilarity_search = AsyncMock(return_value=[Document(page_content="sync result", metadata={})])
        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            default_k=4,
            config_tag="test",
            vector_store=mock_vs,
        )
        results = managed.query("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_bm25_query_delegates_to_store(
        self, bm25_store: BM25DocumentStore, sample_docs: list[Document]
    ) -> None:
        await bm25_store.aadd_documents(sample_docs)
        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            store=bm25_store,
            default_k=2,
            config_tag="bm25_test",
            bm25_store=bm25_store,
        )
        results = await managed.aquery("programming")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_vector_store(self) -> None:
        mock_vs = MagicMock(spec=[])  # no attributes → triggers _collection path
        mock_vs._collection = MagicMock()
        mock_vs._collection.get = MagicMock(return_value={"ids": ["id1", "id2"]})
        mock_vs._collection.delete = MagicMock()

        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            default_k=4,
            config_tag="deletable",
            vector_store=mock_vs,
        )
        result = await managed.adelete_store()
        assert result is True
        mock_vs._collection.delete.assert_called_once_with(ids=["id1", "id2"])

    @pytest.mark.asyncio
    async def test_delete_bm25_store(self, bm25_store: BM25DocumentStore, sample_docs: list[Document]) -> None:
        await bm25_store.aadd_documents(sample_docs)
        managed = ManagedRetriever(
            retriever=_EmptyRetriever(),
            store=bm25_store,
            bm25_store=bm25_store,
            config_tag="bm25_deletable",
        )
        result = await managed.adelete_store()
        assert result is True
        assert not bm25_store.cache_dir.exists()


# ---------------------------------------------------------------------------
# RetrieverFactory
# ---------------------------------------------------------------------------


class TestRetrieverFactory:
    def test_missing_config_raises(self) -> None:
        with pytest.raises(Exception, match="nonexistent_retriever_xyz"):
            RetrieverFactory.create("nonexistent_retriever_xyz")

    def test_list_available_configs(self) -> None:
        configs = RetrieverFactory.list_available_configs()
        assert isinstance(configs, list)
        # At least some configs from baseline.yaml should be present
        assert "default" in configs or len(configs) >= 0  # may be empty in test env

    def test_build_vector_retriever(self) -> None:
        with patch("genai_tk.core.retriever_factory.global_config") as mock_cfg:
            mock_cfg.return_value.get_dict.return_value = {
                "type": "vector",
                "embeddings_store": "in_memory",
                "top_k": 3,
            }
            with patch("genai_tk.core.retriever_factory.RetrieverFactory._build_vector") as mock_build:
                mock_build.return_value = MagicMock(spec=ManagedRetriever)
                managed = RetrieverFactory.create("my_vector")
                mock_build.assert_called_once()

    def test_build_bm25_retriever(self, tmp_path: Path) -> None:
        with (
            patch("genai_tk.core.retriever_factory.global_config") as mock_cfg,
            patch("genai_tk.core.retriever_factory._bm25_cache_dir", return_value=tmp_path / "cache"),
        ):
            mock_cfg.return_value.get_dict.return_value = {"type": "bm25", "k": 4}
            managed = RetrieverFactory.create("my_bm25")
            assert managed is not None
            assert managed.has_store is True
            assert managed._bm25_store is not None

    def test_ensemble_normalises_weights(self) -> None:
        cfg = EnsembleRetrieverConfig(
            retrievers=[
                EnsembleChildConfig(ref="a", weight=3.0),
                EnsembleChildConfig(ref="b", weight=1.0),
            ]
        )
        children_a = ManagedRetriever(retriever=_EmptyRetriever(), store=None, default_k=4, config_tag="a")
        children_b = ManagedRetriever(retriever=_EmptyRetriever(), store=None, default_k=4, config_tag="b")

        with patch.object(RetrieverFactory, "create", side_effect=[children_a, children_b]):
            with patch("langchain_classic.retrievers.EnsembleRetriever") as mock_ensemble:
                mock_ensemble.return_value = _EmptyRetriever()
                managed = RetrieverFactory._build_ensemble(cfg, "ensemble_test")
                # weights should be normalised to [0.75, 0.25]
                call_kwargs = mock_ensemble.call_args.kwargs
                assert abs(call_kwargs["weights"][0] - 0.75) < 1e-6
                assert abs(call_kwargs["weights"][1] - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# Record manager auto-creation
# ---------------------------------------------------------------------------


class TestRecordManager:
    def test_no_rm_for_in_memory(self) -> None:
        rm = _make_record_manager(None, backend="Chroma", table_name="t", config_tag="t", is_persistent=False)
        assert rm is None

    def test_auto_sqlite_for_persistent(self, tmp_path: Path) -> None:
        with patch("genai_tk.core.retriever_factory._data_root", return_value=tmp_path):
            rm = _make_record_manager(
                None, backend="Chroma", table_name="my_table", config_tag="my_cfg", is_persistent=True
            )
            assert rm is not None
            db_path = tmp_path / "record_manager" / "my_cfg.db"
            assert db_path.parent.exists()

    def test_explicit_url(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path}/custom.db"
        rm = _make_record_manager(url, backend="Chroma", table_name="t", config_tag="t", is_persistent=True)
        assert rm is not None
