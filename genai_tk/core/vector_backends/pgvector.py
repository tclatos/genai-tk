"""PgVector vector store backend."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class PgVectorBackend:
    """Creates a PgVector VectorStore with optional hybrid search."""

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        table_name: str,
        config: dict[str, Any],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:

        # EmbeddingsFactory is needed by create_pg_vector_store; reconstruct from the
        # embeddings object's model identity is not straightforward, so we accept the
        # embeddings_factory kwarg when called from EmbeddingsStore.
        raise NotImplementedError(
            "PgVectorBackend.create() must be called via EmbeddingsStore which passes "
            "the EmbeddingsFactory directly. Do not call this method directly."
        )

    @classmethod
    def create_from_factory(
        cls,
        embeddings_factory: Any,
        table_name: str,
        config: dict[str, Any],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:
        from genai_tk.extra.pgvector_factory import PgVectorConfig, create_pg_vector_store

        pg_cfg = PgVectorConfig(
            postgres=config.get("postgres", "default"),
            schema_name=config.get("postgres_schema", "public"),
            metadata_columns=config.get("metadata_columns", []),
            hybrid_search=config.get("hybrid_search", False),
            hybrid_search_config=config.get("hybrid_search_config", {}),
        )
        vector_store, _ = create_pg_vector_store(
            embeddings_factory=embeddings_factory,
            table_name=table_name,
            config=pg_cfg,
        )
        return vector_store
