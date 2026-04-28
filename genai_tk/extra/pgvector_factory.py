"""PgVector store creation and configuration utilities.

Provides a Pydantic-driven factory for creating ``PGVectorStore`` instances
with optional hybrid (vector + full-text) search support.

Usage:
    ```python
    from genai_tk.extra.pgvector_factory import PgVectorConfig, create_pg_vector_store

    cfg = PgVectorConfig(postgres="default", hybrid_search=True)
    vector_store, pg_engine = create_pg_vector_store(
        embeddings_factory=my_embeddings_factory,
        table_name="my_embeddings_ada_002",
        config=cfg,
    )
    ```

References:
    - https://github.com/langchain-ai/langchain-postgres
    - https://python.langchain.com/docs/integrations/vectorstores/pgvectorstore
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import ProgrammingError

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class PgHybridSearchConfig(BaseModel):
    """Configuration for PostgreSQL hybrid (vector + full-text) search."""

    tsv_column: str = "content_tsv"
    tsv_lang: str = "pg_catalog.english"
    fusion_function_parameters: dict[str, Any] = Field(default_factory=dict)
    primary_top_k: int = 4
    secondary_top_k: int = 4
    index_name: str | None = None
    index_type: str = "GIN"


class MetadataColumn(BaseModel):
    """Definition of a custom metadata column in the PgVector table."""

    name: str
    data_type: str


class PgVectorConfig(BaseModel):
    """Configuration for creating a PGVectorStore.

    Attributes:
        postgres: Config tag in the ``postgres:`` YAML section (see ``genai_tk.extra.postgres``).
        schema_name: PostgreSQL schema to use.
        metadata_columns: Extra typed columns to expose in the vector table.
        hybrid_search: Enable hybrid (vector + full-text) search.
        hybrid_search_config: Hybrid search tuning parameters.
    """

    postgres: str = "default"
    schema_name: str = "public"
    metadata_columns: list[MetadataColumn] = Field(default_factory=list)
    hybrid_search: bool = False
    hybrid_search_config: PgHybridSearchConfig = Field(default_factory=PgHybridSearchConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pg_vector_store(
    embeddings_factory: Any,
    table_name: str,
    config: PgVectorConfig,
) -> tuple[VectorStore, Any]:
    """Create and configure a PGVectorStore.

    Args:
        embeddings_factory: ``EmbeddingsFactory`` used to obtain the embedding model
            and vector dimension.
        table_name: Name of the table to create or reuse.
        config: Typed configuration for the store.

    Returns:
        A ``(PGVectorStore, PGEngine)`` tuple.  The ``PGEngine`` is cached in
        ``genai_tk.extra.postgres`` so it is reused for the same config tag.
    """
    try:
        from langchain_postgres import Column, PGEngine, PGVectorStore  # type: ignore[import-untyped]
        from langchain_postgres.v2.hybrid_search_config import (  # type: ignore[import-untyped]
            HybridSearchConfig,
            reciprocal_rank_fusion,
        )
    except ImportError as exc:
        raise ImportError(
            "langchain-postgres is required for PGVectorStore. Install it with: uv add langchain-postgres"
        ) from exc

    from genai_tk.extra.postgres import get_pg_engine

    pg_engine: PGEngine = get_pg_engine(config.postgres)

    # Build hybrid search config if requested
    lc_hybrid_cfg: HybridSearchConfig | None = None
    if config.hybrid_search:
        hsc = config.hybrid_search_config
        lc_hybrid_cfg = HybridSearchConfig(
            tsv_column=hsc.tsv_column,
            tsv_lang=hsc.tsv_lang,
            fts_query="",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters=hsc.fusion_function_parameters,
            primary_top_k=hsc.primary_top_k,
            secondary_top_k=hsc.secondary_top_k,
            index_name=hsc.index_name or f"{table_name}_tsv_index",
            index_type=hsc.index_type,
        )
        logger.debug("Hybrid search enabled, tsv_column='{}'", hsc.tsv_column)

    # Create table (idempotent)
    try:
        pg_engine.init_vectorstore_table(
            table_name=table_name,
            schema_name=config.schema_name,
            vector_size=embeddings_factory.get_dimension(),
            overwrite_existing=False,
            hybrid_search_config=lc_hybrid_cfg,
            metadata_columns=[Column(c.name, c.data_type) for c in config.metadata_columns],
        )
        logger.info(
            "PgVector table ready: {}.{} (hybrid={})",
            config.schema_name,
            table_name,
            config.hybrid_search,
        )
    except ProgrammingError as exc:
        if "already exists" in str(exc).lower():
            logger.debug("Reusing existing PgVector table: {}.{}", config.schema_name, table_name)
        else:
            raise

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=table_name,
        schema_name=config.schema_name,
        embedding_service=embeddings_factory.get(),
        metadata_columns=[c.name for c in config.metadata_columns],
        hybrid_search_config=lc_hybrid_cfg,
    )

    # Apply hybrid search GIN index (async operation — run in a dedicated thread)
    if config.hybrid_search and lc_hybrid_cfg is not None:
        _apply_hybrid_index_sync(vector_store)

    return vector_store, pg_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_hybrid_index_sync(vector_store: Any) -> None:
    """Apply the hybrid search GIN index, safely bridging sync/async.

    ``PGVectorStore.apply_hybrid_search_index()`` is a coroutine.
    This helper runs it in an isolated thread with its own event loop,
    so it works regardless of whether the caller is sync or async.
    """

    async def _apply() -> None:
        await vector_store.apply_hybrid_search_index()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _apply())
            future.result(timeout=30)
        logger.info("Applied hybrid search GIN index on '{}'", vector_store.table_name)
    except Exception as exc:
        logger.warning("Could not apply hybrid search index (non-fatal): {}", exc)
