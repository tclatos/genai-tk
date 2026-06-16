"""Zvec store creation and LangChain adapter utilities."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config


class ZvecVectorStore(VectorStore):
    """LangChain VectorStore adapter for local Zvec collections."""

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
    ) -> None:
        self.embedding = embedding
        self.path = path
        self.collection_name = collection_name
        self.dimension = dimension
        self.vector_field = vector_field
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.metric = metric
        self.optimize_on_add = optimize_on_add
        self._collection = self._open_or_create_collection()

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "ZvecVectorStore":
        """Create a Zvec store from raw texts."""
        path = kwargs.pop("path")
        collection_name = kwargs.pop("collection_name", "zvec_collection")
        dimension = kwargs.pop("dimension", len(embedding.embed_query(texts[0] if texts else "")))
        store = cls(embedding=embedding, path=path, collection_name=collection_name, dimension=dimension, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and insert texts into the Zvec collection."""
        text_list = list(texts)
        if not text_list:
            return []

        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError("metadatas must have the same length as texts")
        if ids is not None and len(ids) != len(text_list):
            raise ValueError("ids must have the same length as texts")

        document_ids = ids or [str(uuid.uuid4()) for _ in text_list]
        vectors = self.embedding.embed_documents(text_list)
        self._insert_embedded_texts(text_list, vectors, metadatas, document_ids)
        return document_ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add LangChain documents to the Zvec collection."""
        ids = kwargs.pop("ids", None)
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return documents most similar to a text query."""
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, **kwargs)

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        """Return documents most similar to a vector."""
        import zvec

        metadata_filter = kwargs.pop("filter", None)
        zvec_filter = metadata_filter if isinstance(metadata_filter, str) else None
        topk = max(k, kwargs.pop("topk", k))
        if isinstance(metadata_filter, dict):
            topk = max(topk, k * 4)

        results = self._collection.query(
            zvec.Query(self.vector_field, vector=embedding),
            topk=topk,
            filter=zvec_filter,
            output_fields=[self.content_field, self.metadata_field],
        )

        documents = [self._doc_from_zvec(result) for result in results]
        if isinstance(metadata_filter, dict):
            documents = [
                document for document in documents if self._metadata_matches(document.metadata, metadata_filter)
            ]
        return documents[:k]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID."""
        if not ids:
            return None
        self._collection.delete(ids=ids)
        return True

    def _open_or_create_collection(self):
        import zvec

        collection_path = Path(self.path)
        collection_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            return zvec.open(str(collection_path))
        except Exception:
            logger.debug(f"Creating Zvec collection at {collection_path}")

        metric_type = self._metric_type(zvec)
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
        return zvec.create_and_open(path=str(collection_path), schema=schema)

    def _insert_embedded_texts(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None,
        ids: list[str],
    ) -> None:
        import zvec

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
                ids, texts, vectors, metadatas or [{}] * len(texts), strict=True
            )
        ]
        statuses = self._collection.insert(docs)
        for status in statuses if isinstance(statuses, list) else [statuses]:
            if hasattr(status, "ok") and not status.ok():
                raise RuntimeError(f"Failed to insert document into Zvec: {status}")

        if self.optimize_on_add:
            self._collection.optimize()

    def _doc_from_zvec(self, zvec_doc: Any) -> Document:
        fields = zvec_doc.fields or {}
        metadata_text = fields.get(self.metadata_field) or "{}"
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError:
            metadata = {"raw_metadata": metadata_text}
        metadata.setdefault("id", zvec_doc.id)
        if getattr(zvec_doc, "score", None) is not None:
            metadata.setdefault("score", zvec_doc.score)
        return Document(page_content=fields.get(self.content_field, ""), metadata=metadata)

    def _metric_type(self, zvec: Any) -> Any:
        metric = self.metric.upper()
        try:
            return getattr(zvec.MetricType, metric)
        except AttributeError as e:
            supported = ", ".join(name for name in dir(zvec.MetricType) if name.isupper())
            raise ValueError(f"Unsupported Zvec metric '{self.metric}'. Supported metrics: {supported}") from e

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in metadata_filter.items())


def create_zvec_vector_store(
    embeddings_factory: Any, table_name: str, config: dict[str, Any], conf: dict[str, Any]
) -> VectorStore:
    """Create and configure a Zvec vector store."""
    try:
        import zvec  # noqa: F401
    except ImportError as e:
        raise ImportError("zvec is required for Zvec vector stores. Install it with: uv add zvec") from e

    storage = config.get("storage") or "zvec"
    if os.path.isabs(storage):
        root_path = Path(storage)
    else:
        try:
            data_root = Path(str(global_config().get_dir_path("paths.data_root", create_if_not_exists=True)))
            root_path = data_root / storage
        except (ValueError, KeyError):
            root_path = Path(storage)

    collection_path = root_path / table_name
    vector_store = ZvecVectorStore(
        embedding=embeddings_factory.get(),
        path=str(collection_path),
        collection_name=config.get("collection_name", table_name),
        dimension=config.get("dimension") or embeddings_factory.get_dimension(),
        vector_field=config.get("vector_field", ZvecVectorStore.DEFAULT_VECTOR_FIELD),
        content_field=config.get("content_field", ZvecVectorStore.DEFAULT_CONTENT_FIELD),
        metadata_field=config.get("metadata_field", ZvecVectorStore.DEFAULT_METADATA_FIELD),
        metric=config.get("metric", "COSINE"),
        optimize_on_add=config.get("optimize_on_add", True),
    )
    conf["path"] = str(collection_path)
    conf["collection_name"] = vector_store.collection_name
    return vector_store
