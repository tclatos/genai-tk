"""PostgreSQL hybrid search retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class PgHybridRetrieverConfig(BaseModel):
    """Configuration for a PostgreSQL hybrid (vector + full-text) retriever."""

    embeddings: str = "default"
    table_name_prefix: str = "embeddings"
    postgres: str = "default"
    schema_name: str = "public"
    top_k: int = 4
    metadata_columns: list[dict[str, str]] = Field(default_factory=list)
    hybrid_search: bool = True
    hybrid_search_config: dict[str, Any] = Field(default_factory=dict)
    record_manager_url: str | None = None


class PgHybridRetriever:
    """Builder for PostgreSQL hybrid search retrievers."""

    config_model = PgHybridRetrieverConfig

    @classmethod
    def build(cls, cfg: PgHybridRetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from genai_tk.core.embeddings_factory import EmbeddingsFactory
        from genai_tk.core.retriever_factory import ManagedRetriever, VectorDocumentStore, _make_record_manager
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
