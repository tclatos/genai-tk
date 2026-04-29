"""Retriever Factory — composable, YAML-configured RAG retrievers.

Provides a unified ``ManagedRetriever`` that wraps a LangChain ``BaseRetriever``
with optional document ingestion support, and a ``RetrieverFactory`` that builds
any retriever type from a YAML config tag.

Supported retriever types
    ``vector``         — simple vector-similarity via EmbeddingsStore
    ``bm25``           — BM25 full-text retrieval, persisted to disk
    ``pg_hybrid``      — PostgreSQL vector + full-text (PGVectorStore)
    ``ensemble``       — recursive composition (vector + BM25, etc.)
    ``reranked``       — any retriever wrapped with a reranking step
    ``zero_entropy``   — ZeroEntropy SDK retriever

YAML configuration example::

    retrievers:
      default:
        type: vector
        embeddings_store: in_memory_chroma
        top_k: 4

      bm25_local:
        type: bm25
        k: 4

      hybrid_ensemble:
        type: ensemble
        retrievers:
          - ref: default
            weight: 0.7
          - ref: bm25_local
            weight: 0.3

      pg_hybrid:
        type: pg_hybrid
        embeddings: default
        postgres: default
        hybrid_search: true

      reranked_default:
        type: reranked
        retriever: hybrid_ensemble
        reranker: embeddings
        top_k: 3

Usage::

    managed = RetrieverFactory.create("hybrid_ensemble")

    # Async (preferred)
    docs = await managed.aquery("my question", k=5)
    await managed.aadd_documents(my_docs)

    # Sync wrappers (CLI / notebooks)
    docs = managed.query("my question")
    managed.add_documents(my_docs)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# Config models (Pydantic v2, discriminated union)
# ---------------------------------------------------------------------------


class VectorRetrieverConfig(BaseModel):
    """Simple vector-similarity retriever backed by an EmbeddingsStore."""

    type: Literal["vector"] = "vector"
    embeddings_store: str
    top_k: int = 4
    search_type: str = "similarity"
    record_manager_url: str | None = None


class BM25RetrieverConfig(BaseModel):
    """BM25 full-text retriever with optional Spacy preprocessing."""

    type: Literal["bm25"] = "bm25"
    k: int = 4
    preprocessing: str = "default"
    spacy_model: str = "en_core_web_sm"
    cache_dir: str | None = None
    bm25_params: dict[str, Any] = Field(default_factory=dict)


class PgHybridRetrieverConfig(BaseModel):
    """PostgreSQL hybrid search (vector + full-text via PGVectorStore)."""

    type: Literal["pg_hybrid"] = "pg_hybrid"
    embeddings: str = "default"
    table_name_prefix: str = "embeddings"
    postgres: str = "default"
    schema_name: str = "public"
    top_k: int = 4
    metadata_columns: list[dict[str, str]] = Field(default_factory=list)
    hybrid_search: bool = True
    hybrid_search_config: dict[str, Any] = Field(default_factory=dict)
    record_manager_url: str | None = None


class EnsembleChildConfig(BaseModel):
    """One child retriever inside an ensemble."""

    ref: str
    weight: float = 0.5


class EnsembleRetrieverConfig(BaseModel):
    """Recursive composition of multiple retrievers with weighted fusion."""

    type: Literal["ensemble"] = "ensemble"
    retrievers: list[EnsembleChildConfig]


class RerankedRetrieverConfig(BaseModel):
    """Any retriever wrapped with a reranking / compression step."""

    type: Literal["reranked"] = "reranked"
    retriever: str
    reranker: str = "embeddings"
    top_k: int = 3
    fetch_k: int = 10
    reranker_model: str | None = None
    embeddings: str | None = None


class ZeroEntropyRetrieverConfig(BaseModel):
    """ZeroEntropy SDK retriever (read-only, external service)."""

    type: Literal["zero_entropy"] = "zero_entropy"
    collection_name: str
    k: int = 5
    retrieval_type: str = "documents"


RetrieverConfig = Annotated[
    Union[
        VectorRetrieverConfig,
        BM25RetrieverConfig,
        PgHybridRetrieverConfig,
        EnsembleRetrieverConfig,
        RerankedRetrieverConfig,
        ZeroEntropyRetrieverConfig,
    ],
    Field(discriminator="type"),
]


def _parse_retriever_config(data: dict[str, Any]) -> RetrieverConfig:
    from pydantic import TypeAdapter

    return TypeAdapter(RetrieverConfig).validate_python(data)


# ---------------------------------------------------------------------------
# Document stores (for ingestion)
# ---------------------------------------------------------------------------


class VectorDocumentStore:
    """Ingestion store backed by a LangChain VectorStore."""

    def __init__(self, vector_store: Any, record_manager: Any | None = None, source_id_key: str = "source") -> None:
        self.vector_store = vector_store
        self.record_manager = record_manager
        self.source_id_key = source_id_key

    async def aadd_documents(self, docs: list[Document]) -> Any:
        if self.record_manager is not None:
            from langchain_classic.indexes import index

            loop = asyncio.get_running_loop()
            rm = self.record_manager
            vs = self.vector_store
            sid = self.source_id_key
            return await loop.run_in_executor(
                None,
                lambda: index(
                    docs,
                    rm,
                    vs,
                    cleanup="incremental",
                    source_id_key=sid,
                    key_encoder="blake2b",
                ),
            )
        return await self.vector_store.aadd_documents(docs)


class BM25DocumentStore:
    """Ingestion + retrieval store backed by a BM25 index persisted to disk.

    Following the same pattern as LangChain's ``BaseRetriever``:
    sync methods are the primary implementation; async methods delegate to
    them via ``run_in_executor`` so they never block the event loop.
    Notebooks and CLI call the sync methods directly.
    """

    def __init__(
        self,
        cache_dir: Path,
        preprocessing: str = "default",
        spacy_model: str = "en_core_web_sm",
        bm25_params: dict[str, Any] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.preprocessing = preprocessing
        self.spacy_model = spacy_model
        self.bm25_params = bm25_params or {}
        self._current_retriever: Any | None = None

    # -- preprocessing -------------------------------------------------------

    def _preprocess_func(self) -> Any:
        from genai_tk.extra.retrievers.bm25s_retriever import default_preprocessing_func

        if self.preprocessing == "spacy":
            from genai_tk.extra.retrievers.bm25s_retriever import get_spacy_preprocess_fn

            return get_spacy_preprocess_fn(self.spacy_model)
        return default_preprocessing_func

    # -- sync core -----------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Build (or rebuild) the BM25 index from *docs* and persist it to disk."""
        from genai_tk.extra.retrievers.bm25s_retriever import BM25FastRetriever

        index_path = self.cache_dir / "bm25_index"
        docs_path = self.cache_dir / "documents.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._current_retriever = BM25FastRetriever.from_documents(
            documents=list(docs),
            preprocess_func=self._preprocess_func(),
            cache_dir=index_path,
            bm25_params=self.bm25_params,
        )
        doc_data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        docs_path.write_text(json.dumps(doc_data, ensure_ascii=False))
        return [str(i) for i in range(len(docs))]

    def get_or_load_retriever(self, k: int = 4) -> Any | None:
        """Return the in-memory retriever, loading from disk cache if necessary."""
        if self._current_retriever is not None:
            self._current_retriever.k = k
            return self._current_retriever

        from genai_tk.extra.retrievers.bm25s_retriever import BM25FastRetriever

        index_path = self.cache_dir / "bm25_index"
        docs_path = self.cache_dir / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            return None
        try:
            doc_data = json.loads(docs_path.read_text())
            loaded_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in doc_data]
            retriever = BM25FastRetriever.from_index_file(
                index_file=index_path,
                preprocess_func=self._preprocess_func(),
                k=k,
            )
            # documents.json preserves full metadata; override the retriever's plain-text docs
            retriever.docs = loaded_docs
            self._current_retriever = retriever
            return retriever
        except Exception as exc:
            logger.warning("Failed to load BM25 index from {}: {}", index_path, exc)
            return None

    def get_relevant_documents(self, query: str, k: int = 4) -> list[Document]:
        """Return BM25-ranked documents for *query* (sync)."""
        retriever = self.get_or_load_retriever(k=k)
        if retriever is None:
            logger.warning("BM25 index not yet built. Call add_documents() first.")
            return []
        return retriever.invoke(query)

    # -- async wrappers (delegate to sync via executor) ----------------------

    async def aadd_documents(self, docs: list[Document]) -> list[str]:
        """Async wrapper for :meth:`add_documents`. Runs in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.add_documents, list(docs))

    async def aget_relevant_documents(self, query: str, k: int = 4) -> list[Document]:
        """Async wrapper for :meth:`get_relevant_documents`. Runs in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_relevant_documents, query, k)


class CompositeDocumentStore:
    """Fans out ingestion to multiple document stores concurrently."""

    def __init__(self, stores: list[VectorDocumentStore | BM25DocumentStore]) -> None:
        self.stores = stores

    async def aadd_documents(self, docs: list[Document]) -> Any:
        results = await asyncio.gather(*[s.aadd_documents(docs) for s in self.stores])
        return results[0] if results else []


# ---------------------------------------------------------------------------
# ManagedRetriever
# ---------------------------------------------------------------------------


class ManagedRetriever:
    """Async-first retriever with optional document store for ingestion.

    The primary API is ``aquery`` / ``aadd_documents``.
    Sync wrappers ``query`` / ``add_documents`` are provided for CLI and
    notebook use — they must be called from a non-async context.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        store: VectorDocumentStore | BM25DocumentStore | CompositeDocumentStore | None = None,
        default_k: int = 4,
        config_tag: str = "",
        *,
        vector_store: Any | None = None,
        bm25_store: BM25DocumentStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.store = store
        self.default_k = default_k
        self.config_tag = config_tag
        self._vector_store = vector_store
        self._bm25_store = bm25_store

    # -- properties ----------------------------------------------------------

    @property
    def has_store(self) -> bool:
        """Return True if this retriever supports document ingestion."""
        return self.store is not None

    # -- async API -----------------------------------------------------------

    async def aquery(self, query: str, k: int | None = None, filter: dict[str, Any] | None = None) -> list[Document]:
        """Return relevant documents for *query*.

        Args:
            query: Search string.
            k: Number of results; falls back to ``default_k`` when ``None``.
            filter: Metadata filter dict (only supported for vector-store backends).

        Returns:
            List of matching ``Document`` instances.
        """
        effective_k = k if k is not None else self.default_k

        # BM25: delegate to the store so it reflects the latest rebuilt index
        if self._bm25_store is not None:
            return await self._bm25_store.aget_relevant_documents(query, k=effective_k)

        # Vector store: direct access supports k + filter overrides
        if self._vector_store is not None:
            return await self._vector_store.asimilarity_search(query, k=effective_k, filter=filter)

        # Generic retriever (ensemble, reranked, zero_entropy …)
        return await self.retriever.ainvoke(query)

    async def aadd_documents(self, docs: list[Document]) -> Any:
        """Ingest *docs* into the underlying store.

        Raises:
            RuntimeError: If the retriever has no document store configured.
        """
        if self.store is None:
            raise RuntimeError(
                f"Retriever '{self.config_tag}' is read-only — no document store configured. "
                "ZeroEntropy and reranked retrievers without a writable base are read-only."
            )
        return await self.store.aadd_documents(list(docs))

    async def adelete_store(self) -> bool:
        """Delete/clear all stored documents.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if self._vector_store is not None:
            try:
                if hasattr(self._vector_store, "delete_collection"):
                    self._vector_store.delete_collection()
                    return True
                # Chroma-style: delete all document IDs
                all_docs = self._vector_store._collection.get()  # type: ignore[attr-defined]
                if all_docs and all_docs.get("ids"):
                    self._vector_store._collection.delete(ids=all_docs["ids"])  # type: ignore[attr-defined]
                return True
            except Exception as exc:
                logger.error("Failed to delete vector store '{}': {}", self.config_tag, exc)
                return False

        if self._bm25_store is not None:
            import shutil

            try:
                shutil.rmtree(self._bm25_store.cache_dir, ignore_errors=True)
                self._bm25_store._current_retriever = None
                return True
            except Exception as exc:
                logger.error("Failed to delete BM25 cache '{}': {}", self.config_tag, exc)
                return False

        raise NotImplementedError(
            f"adelete_store() is not supported for retriever type '{self.config_tag}'. "
            "Only vector and BM25 backed retrievers support deletion."
        )

    # -- sync wrappers (CLI / notebooks) ------------------------------------

    def query(self, query: str, k: int | None = None, filter: dict[str, Any] | None = None) -> list[Document]:
        """Sync wrapper for :meth:`aquery`. Call only from non-async contexts."""
        return asyncio.run(self.aquery(query, k=k, filter=filter))

    def add_documents(self, docs: list[Document]) -> Any:
        """Sync wrapper for :meth:`aadd_documents`. Call only from non-async contexts."""
        return asyncio.run(self.aadd_documents(docs))

    def delete_store(self) -> bool:
        """Sync wrapper for :meth:`adelete_store`. Call only from non-async contexts."""
        return asyncio.run(self.adelete_store())

    # -- introspection -------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return a summary dict describing this retriever."""
        stats: dict[str, Any] = {"config_tag": self.config_tag, "default_k": self.default_k}
        if self._vector_store is not None:
            stats["vector_backend"] = type(self._vector_store).__name__
        if self._bm25_store is not None:
            stats["bm25_cache_dir"] = str(self._bm25_store.cache_dir)
        return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    try:
        return Path(global_config().get_str("paths.data_root"))
    except Exception:
        return Path("data")


def _bm25_cache_dir(cache_dir_cfg: str | None, config_tag: str) -> Path:
    if cache_dir_cfg:
        return Path(cache_dir_cfg)
    return _data_root() / "bm25_cache" / config_tag


def _is_persistent(es: Any) -> bool:
    if es.backend == "Chroma":
        return es.config.get("storage", "::memory::") != "::memory::"
    return es.backend == "PgVector"


def _make_record_manager(
    rm_url: str | None,
    *,
    backend: str,
    table_name: str,
    config_tag: str,
    is_persistent: bool,
) -> Any | None:
    """Build a SQLRecordManager, auto-generating a SQLite path for persistent stores."""
    if not is_persistent and rm_url is None:
        return None

    from langchain_classic.indexes import SQLRecordManager

    if rm_url is None:
        db_path = _data_root() / "record_manager" / f"{config_tag}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        rm_url = f"sqlite:///{db_path}"

    rm = SQLRecordManager(f"{backend}/{table_name}", db_url=rm_url)
    rm.create_schema()
    return rm


class _EmptyRetriever(BaseRetriever):
    """Placeholder retriever returned when a BM25 index has not yet been built."""

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return []


# ---------------------------------------------------------------------------
# RetrieverFactory
# ---------------------------------------------------------------------------


class RetrieverFactory:
    """Factory that creates :class:`ManagedRetriever` instances from YAML config tags."""

    @classmethod
    def create(cls, config_tag: str) -> ManagedRetriever:
        """Build a ``ManagedRetriever`` from the named configuration.

        Args:
            config_tag: Key in the ``retrievers`` YAML section.

        Returns:
            Configured ``ManagedRetriever`` ready for use.
        """
        try:
            raw = global_config().get_dict(f"retrievers.{config_tag}")
        except (ValueError, KeyError) as exc:
            available = cls.list_available_configs()
            raise ValueError(f"Retriever configuration '{config_tag}' not found. Available: {available}") from exc

        cfg = _parse_retriever_config(raw)
        return cls._build(cfg, config_tag)

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List all retriever config tags available in YAML."""
        try:
            cfgs = global_config().get("retrievers", {})
            return list(cfgs.keys()) if hasattr(cfgs, "keys") else []
        except Exception:
            return []

    # -- dispatch ------------------------------------------------------------

    @classmethod
    def _build(cls, cfg: Any, config_tag: str) -> ManagedRetriever:
        builders: dict[type, Any] = {
            VectorRetrieverConfig: cls._build_vector,
            BM25RetrieverConfig: cls._build_bm25,
            PgHybridRetrieverConfig: cls._build_pg_hybrid,
            EnsembleRetrieverConfig: cls._build_ensemble,
            RerankedRetrieverConfig: cls._build_reranked,
            ZeroEntropyRetrieverConfig: cls._build_zero_entropy,
        }
        builder = builders.get(type(cfg))
        if builder is None:
            raise ValueError(f"Unsupported retriever config type: {type(cfg)}")
        return builder(cfg, config_tag)

    # -- vector --------------------------------------------------------------

    @classmethod
    def _build_vector(cls, cfg: VectorRetrieverConfig, config_tag: str) -> ManagedRetriever:
        from genai_tk.core.embeddings_store import EmbeddingsStore

        es = EmbeddingsStore.create_from_config(cfg.embeddings_store)
        vs = es.get_vector_store()
        rm = _make_record_manager(
            cfg.record_manager_url,
            backend=es.backend or "unknown",
            table_name=es.table_name,
            config_tag=config_tag,
            is_persistent=_is_persistent(es),
        )
        retriever = vs.as_retriever(search_kwargs={"k": cfg.top_k})
        store = VectorDocumentStore(vector_store=vs, record_manager=rm)
        return ManagedRetriever(
            retriever=retriever,
            store=store,
            default_k=cfg.top_k,
            config_tag=config_tag,
            vector_store=vs,
        )

    # -- bm25 ----------------------------------------------------------------

    @classmethod
    def _build_bm25(cls, cfg: BM25RetrieverConfig, config_tag: str) -> ManagedRetriever:
        cache_dir = _bm25_cache_dir(cfg.cache_dir, config_tag)
        bm25_store = BM25DocumentStore(
            cache_dir=cache_dir,
            preprocessing=cfg.preprocessing,
            spacy_model=cfg.spacy_model,
            bm25_params=cfg.bm25_params,
        )
        retriever = bm25_store.get_or_load_retriever(k=cfg.k) or _EmptyRetriever()
        return ManagedRetriever(
            retriever=retriever,
            store=bm25_store,
            default_k=cfg.k,
            config_tag=config_tag,
            bm25_store=bm25_store,
        )

    # -- pg_hybrid -----------------------------------------------------------

    @classmethod
    def _build_pg_hybrid(cls, cfg: PgHybridRetrieverConfig, config_tag: str) -> ManagedRetriever:
        from genai_tk.core.embeddings_factory import EmbeddingsFactory
        from genai_tk.extra.pgvector_factory import (
            MetadataColumn,
            PgHybridSearchConfig,
            PgVectorConfig,
            create_pg_vector_store,
        )

        ef = EmbeddingsFactory(embeddings=cfg.embeddings)
        table_name = f"{cfg.table_name_prefix}_{ef.short_name()}"
        pg_cfg = PgVectorConfig(
            postgres=cfg.postgres,
            schema_name=cfg.schema_name,
            metadata_columns=[MetadataColumn(**c) for c in cfg.metadata_columns],
            hybrid_search=cfg.hybrid_search,
            hybrid_search_config=PgHybridSearchConfig(**cfg.hybrid_search_config)
            if cfg.hybrid_search_config
            else PgHybridSearchConfig(),
        )
        vs, _ = create_pg_vector_store(embeddings_factory=ef, table_name=table_name, config=pg_cfg)
        rm = _make_record_manager(
            cfg.record_manager_url,
            backend="PgVector",
            table_name=table_name,
            config_tag=config_tag,
            is_persistent=True,
        )
        retriever = vs.as_retriever(search_kwargs={"k": cfg.top_k})
        store = VectorDocumentStore(vector_store=vs, record_manager=rm)
        return ManagedRetriever(
            retriever=retriever,
            store=store,
            default_k=cfg.top_k,
            config_tag=config_tag,
            vector_store=vs,
        )

    # -- ensemble ------------------------------------------------------------

    @classmethod
    def _build_ensemble(cls, cfg: EnsembleRetrieverConfig, config_tag: str) -> ManagedRetriever:
        from langchain_classic.retrievers import EnsembleRetriever

        children = [cls.create(child.ref) for child in cfg.retrievers]
        weights = [child.weight for child in cfg.retrievers]
        total = sum(weights) or 1.0
        normalised = [w / total for w in weights]

        ensemble = EnsembleRetriever(
            retrievers=[c.retriever for c in children],
            weights=normalised,
        )
        stores = [c.store for c in children if c.store is not None]
        composite = CompositeDocumentStore(stores=stores) if stores else None

        return ManagedRetriever(
            retriever=ensemble,
            store=composite,
            default_k=4,
            config_tag=config_tag,
        )

    # -- reranked ------------------------------------------------------------

    @classmethod
    def _build_reranked(cls, cfg: RerankedRetrieverConfig, config_tag: str) -> ManagedRetriever:
        from langchain_classic.retrievers import ContextualCompressionRetriever

        base = cls.create(cfg.retriever)
        compressor = _build_compressor(cfg)
        reranked = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base.retriever,
        )
        return ManagedRetriever(
            retriever=reranked,
            store=base.store,
            default_k=cfg.top_k,
            config_tag=config_tag,
            vector_store=base._vector_store,
            bm25_store=base._bm25_store,
        )

    # -- zero_entropy --------------------------------------------------------

    @classmethod
    def _build_zero_entropy(cls, cfg: ZeroEntropyRetrieverConfig, config_tag: str) -> ManagedRetriever:
        from genai_tk.extra.retrievers.zeroentropy import ZeroEntropyConfig, ZeroEntropyRetriever

        ze_cfg = ZeroEntropyConfig(
            collection_name=cfg.collection_name,
            k=cfg.k,
            retrieval_type=cfg.retrieval_type,
        )
        return ManagedRetriever(
            retriever=ZeroEntropyRetriever(config=ze_cfg),
            store=None,
            default_k=cfg.k,
            config_tag=config_tag,
        )


# ---------------------------------------------------------------------------
# Compressor / reranker builders
# ---------------------------------------------------------------------------


def _build_compressor(cfg: RerankedRetrieverConfig) -> Any:
    """Build a LangChain document compressor from a RerankedRetrieverConfig."""
    if cfg.reranker == "cohere":
        try:
            from langchain_cohere import CohereRerank  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("langchain-cohere required. Install: uv add langchain-cohere") from exc
        return CohereRerank(model=cfg.reranker_model or "rerank-english-v3.0", top_n=cfg.top_k)

    if cfg.reranker == "cross_encoder":
        try:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "langchain-community required for cross-encoder. Install: uv add langchain-community"
            ) from exc
        model = HuggingFaceCrossEncoder(model_name=cfg.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoderReranker(model=model, top_n=cfg.top_k)

    if cfg.reranker == "embeddings":
        from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

        from genai_tk.core.embeddings_factory import EmbeddingsFactory

        ef = EmbeddingsFactory(embeddings=cfg.embeddings) if cfg.embeddings else EmbeddingsFactory()
        return EmbeddingsFilter(embeddings=ef.get(), similarity_threshold=0.7)

    raise ValueError(f"Unknown reranker '{cfg.reranker}'. Choose from: 'embeddings', 'cohere', 'cross_encoder'.")
