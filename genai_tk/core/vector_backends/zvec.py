"""Persistent Zvec stores with bounded native-handle and file-lock lifetimes."""

from __future__ import annotations

import heapq
import json
import math
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field, JsonValue, TypeAdapter
from zvec import Collection, Doc, Status

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory

_METADATA_ADAPTER = TypeAdapter(dict[str, JsonValue])


class ZvecConfig(BaseModel):
    """Configuration for a new or existing Zvec index."""

    storage: str = "vector_store_zvec"
    collection_name: str | None = None
    dimension: int | None = Field(default=None, gt=0)
    vector_field: str = "embedding"
    content_field: str = "page_content"
    metadata_field: str = "metadata"
    metric: str = "COSINE"
    optimize_on_add: bool = True
    lock_timeout: float = Field(default=30.0, gt=0)

    model_config = {"extra": "forbid"}


class ZvecVectorStore(VectorStore):
    """Persist local vector collections without retaining native file locks."""

    index_key_encoder: ClassVar[str] = "sha256"

    DEFAULT_VECTOR_FIELD: ClassVar[str] = "embedding"
    DEFAULT_CONTENT_FIELD: ClassVar[str] = "page_content"
    DEFAULT_METADATA_FIELD: ClassVar[str] = "metadata"

    def __init__(
        self,
        embedding: Embeddings,
        path: str,
        collection_name: str,
        dimension: int,
        vector_field: str = DEFAULT_VECTOR_FIELD,
        content_field: str = DEFAULT_CONTENT_FIELD,
        metadata_field: str = DEFAULT_METADATA_FIELD,
        metric: str = "COSINE",
        optimize_on_add: bool = True,
        lock_timeout: float = 30.0,
    ) -> None:
        self.embedding = embedding
        self.path = str(Path(path).resolve())
        self.collection_name = collection_name
        self.dimension = dimension
        self.vector_field = vector_field
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.metric = metric
        self.optimize_on_add = optimize_on_add
        self.lock_timeout = lock_timeout
        with self._collection_context():
            pass

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ZvecVectorStore:
        """Create a persistent store and ingest texts."""
        path = kwargs.pop("path")
        collection_name = kwargs.pop("collection_name", "zvec_collection")
        dimension = kwargs.pop("dimension", None) or len(embedding.embed_query(texts[0] if texts else ""))
        store = cls(embedding=embedding, path=path, collection_name=collection_name, dimension=dimension, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @contextmanager
    def _collection_context(self, *, write: bool = False) -> Iterator[Collection]:
        import zvec

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A separate lock serializes factories, threads, and processes before
        # Zvec obtains its own native lock. Never recover an open error by
        # creating over an existing directory (which may contain legacy data).
        with FileLock(str(path.with_name(path.name + ".lock")), timeout=self.lock_timeout):
            collection = zvec.open(str(path)) if path.exists() else self._create_collection()
            try:
                yield collection
                if write:
                    collection.flush()
            finally:
                collection.close()

    def _create_collection(self) -> Collection:
        import zvec

        try:
            metric_type = getattr(zvec.MetricType, self.metric.upper())
        except AttributeError as exc:
            raise ValueError(f"Unsupported Zvec metric '{self.metric}'") from exc
        schema = zvec.CollectionSchema(
            name=self.collection_name,
            fields=[
                zvec.FieldSchema(name=self.content_field, data_type=zvec.DataType.STRING, nullable=True),
                zvec.FieldSchema(name=self.metadata_field, data_type=zvec.DataType.STRING, nullable=True),
            ],
            vectors=[
                zvec.VectorSchema(
                    name=self.vector_field,
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=self.dimension,
                    index_param=zvec.HnswIndexParam(metric_type=metric_type),
                )
            ],
        )
        return zvec.create_and_open(path=self.path, schema=schema)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert texts, flushing them before releasing the lock."""
        import zvec

        text_list = list(texts)
        if not text_list:
            return []
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError("metadatas must have the same length as texts")
        if ids is not None and len(ids) != len(text_list):
            raise ValueError("ids must have the same length as texts")
        document_ids = ids or [str(uuid.uuid4()) for _ in text_list]
        vectors = self.embedding.embed_documents(text_list)
        docs = [
            zvec.Doc(
                id=document_id,
                vectors={self.vector_field: vector},
                fields={
                    self.content_field: text,
                    self.metadata_field: json.dumps(metadata or {}),
                },
            )
            for document_id, text, vector, metadata in zip(
                document_ids, text_list, vectors, metadatas or [{}] * len(text_list), strict=True
            )
        ]
        with self._collection_context(write=True) as collection:
            self._check_statuses(collection.upsert(docs))
            if self.optimize_on_add:
                collection.optimize()
        return document_ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Persist documents with their metadata."""
        ids = kwargs.pop("ids", None)
        if ids is None and all(document.id is not None for document in documents):
            ids = [document.id for document in documents]
        return self.add_texts(
            [document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
            ids=ids,
            **kwargs,
        )

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Find similar documents using the configured embeddings model."""
        return self.similarity_search_by_vector(self.embedding.embed_query(query), k=k, **kwargs)

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        """Search vectors with native expressions or equality metadata filters."""
        import zvec

        if k < 1:
            return []
        metadata_filter = kwargs.pop("filter", None)
        if metadata_filter is not None and not isinstance(metadata_filter, (str, dict)):
            raise ValueError("Zvec filter must be an expression string or an equality dictionary")
        with self._collection_context() as collection:
            if not collection.stats.doc_count:
                return []
            if isinstance(metadata_filter, dict) and metadata_filter:
                # Stream and rank equality matches with O(k) retained vectors;
                # native queries cannot request more than 100,000 neighbors.
                metric = collection.schema.vector(self.vector_field).index_param.metric_type.name
                with collection.iter_docs(include_vector=True) as docs:
                    matches = (doc for doc in docs if self._matches_filter(doc, metadata_filter))
                    ranked = heapq.nlargest(
                        k,
                        matches,
                        key=lambda doc: self._similarity_score(embedding, doc.vector(self.vector_field), metric),
                    )
                return [self._doc_from_zvec(doc) for doc in ranked]
            topk = max(k, kwargs.pop("topk", k))
            results = collection.query(
                zvec.Query(self.vector_field, vector=embedding),
                topk=topk,
                filter=metadata_filter if isinstance(metadata_filter, str) else None,
                output_fields=[self.content_field, self.metadata_field],
            )
            documents = [self._doc_from_zvec(result) for result in results]
        return documents[:k]

    def _matches_filter(self, doc: Doc, expected: dict[str, object]) -> bool:
        metadata = self._doc_from_zvec(doc).metadata
        return all(metadata.get(key) == value for key, value in expected.items())

    @staticmethod
    def _similarity_score(query: Sequence[float], vector: Sequence[float], metric: str) -> float:
        if metric == "L2":
            return -math.fsum((left - right) ** 2 for left, right in zip(query, vector, strict=True))
        dot = math.fsum(left * right for left, right in zip(query, vector, strict=True))
        if metric == "IP":
            return dot
        if metric == "COSINE":
            norm = math.sqrt(math.fsum(value * value for value in query) * math.fsum(value * value for value in vector))
            return dot / norm if norm else 0.0
        raise ValueError(f"Equality filters do not support Zvec metric '{metric}'")

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Read persisted documents by ID."""
        if not ids:
            return []
        with self._collection_context() as collection:
            docs = collection.fetch(list(ids), include_vector=False)
            return [self._doc_from_zvec(docs[doc_id]) for doc_id in ids if doc_id in docs]

    def iter_documents(self) -> Iterator[Document]:
        """Return a document snapshot after closing the native iterator."""
        with self._collection_context() as collection:
            with collection.iter_docs(include_vector=False) as docs:
                snapshot = [self._doc_from_zvec(doc) for doc in docs]
        return iter(snapshot)

    def document_count(self) -> int:
        """Count persisted documents."""
        with self._collection_context() as collection:
            return collection.stats.doc_count

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete only the explicitly supplied document IDs."""
        if not ids:
            return False
        with self._collection_context(write=True) as collection:
            self._check_statuses(collection.delete(ids=ids))
        return True

    def delete_collection(self) -> None:
        """Clear documents while preserving the collection's storage and schema."""
        with self._collection_context(write=True) as collection:
            with collection.iter_docs(include_vector=False) as docs:
                ids = [doc.id for doc in docs]
            if ids:
                self._check_statuses(collection.delete(ids=ids))

    @staticmethod
    def _check_statuses(statuses: Status | list[Status]) -> None:
        for status in statuses if isinstance(statuses, list) else [statuses]:
            if not status.ok():
                raise RuntimeError(f"Zvec document operation failed: {status}")

    def _doc_from_zvec(self, doc: Doc) -> Document:
        fields = doc.fields or {}
        metadata = _METADATA_ADAPTER.validate_json(fields.get(self.metadata_field) or "{}")
        return Document(id=doc.id, page_content=fields.get(self.content_field) or "", metadata=metadata)


class ZvecBackend:
    """Create persistent Zvec stores through the configured embeddings factory."""

    @classmethod
    def create_from_factory(
        cls,
        embeddings_factory: EmbeddingsFactory,
        table_name: str,
        config: dict[str, object],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:
        cfg = ZvecConfig.model_validate(config)
        if cfg.storage == "::memory::":
            raise ValueError("Zvec requires persistent storage; use InMemoryBackend for ephemeral work")
        root = Path(cfg.storage)
        if not root.is_absolute():
            root = Path(global_config().get_str("paths.data_root")) / root
        return ZvecVectorStore(
            embedding=embeddings_factory.get(),
            path=str(root / table_name),
            collection_name=cfg.collection_name or table_name,
            dimension=cfg.dimension or embeddings_factory.get_dimension(),
            vector_field=cfg.vector_field,
            content_field=cfg.content_field,
            metadata_field=cfg.metadata_field,
            metric=cfg.metric,
            optimize_on_add=cfg.optimize_on_add,
            lock_timeout=cfg.lock_timeout,
        )


def create_zvec_vector_store(
    embeddings_factory: EmbeddingsFactory, table_name: str, config: dict[str, object], conf: dict[str, object]
) -> VectorStore:
    """Preserve the legacy helper's path defaults and collection naming."""
    options = {**config, "storage": config.get("storage") or "zvec"}
    store = ZvecBackend.create_from_factory(embeddings_factory, table_name, options)
    conf["path"] = store.path
    conf["collection_name"] = store.collection_name
    return store
