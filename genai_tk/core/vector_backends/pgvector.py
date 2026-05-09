"""PostgreSQL vector store backend — connection management and PGVectorStore factory.

Supports both external PostgreSQL servers and embedded PostgreSQL via pgembed.
Provides the ``PgVectorBackend`` class used by ``EmbeddingsStore``, plus the
lower-level helpers ``get_pg_engine`` / ``get_postgres_url`` for direct access.

Configuration examples:
    ```yaml
    postgres:
      default:
        mode: external
        url: postgresql+asyncpg://${oc.env:POSTGRES_USER}:${oc.env:POSTGRES_PASSWORD}@localhost:5432/db

      embedded:
        mode: pgembed
        data_dir: ${paths.data_root}/pgembed
        extensions: [vector]
    ```

Usage:
    ```python
    from genai_tk.core.vector_backends.pgvector import (
        get_pg_engine,
        get_postgres_url,
        PgVectorConfig,
        create_pg_vector_store,
        PgVectorBackend,
    )

    engine = get_pg_engine("default")
    cfg = PgVectorConfig(postgres="default", hybrid_search=True)
    vector_store, _ = create_pg_vector_store(embeddings_factory, "my_table", cfg)
    ```
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import ProgrammingError

from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# PostgreSQL connection management
# ---------------------------------------------------------------------------


class PostgresConfig(BaseModel):
    """Configuration for a PostgreSQL connection source."""

    mode: str = "external"
    url: str | None = None
    data_dir: str | None = None
    extensions: list[str] = Field(default_factory=lambda: ["vector"])


_engine_cache: dict[str, Any] = {}


def get_pg_engine(config_tag: str = "default") -> Any:
    """Return a cached PGEngine for the given config tag.

    Args:
        config_tag: Key in the ``postgres`` YAML section.

    Returns:
        Configured ``PGEngine`` instance from langchain-postgres.
    """
    if config_tag in _engine_cache:
        return _engine_cache[config_tag]

    try:
        from langchain_postgres import PGEngine
    except ImportError as exc:
        raise ImportError(
            "langchain-postgres is required for PGEngine. Install it with: uv add langchain-postgres"
        ) from exc

    url = get_postgres_url(config_tag)
    engine = PGEngine.from_connection_string(url=url)
    _engine_cache[config_tag] = engine
    logger.debug("Created PGEngine for postgres config tag '{}'", config_tag)
    return engine


def invalidate_engine_cache(config_tag: str | None = None) -> None:
    """Remove cached engine(s), forcing re-creation on the next call.

    Args:
        config_tag: Tag to invalidate. Clears all cached engines when ``None``.
    """
    if config_tag is None:
        _engine_cache.clear()
    else:
        _engine_cache.pop(config_tag, None)


def get_postgres_url(config_tag: str = "default") -> str:
    """Resolve the PostgreSQL connection URL for the given config tag.

    For ``mode: pgembed``, the embedded server is started automatically on first
    call. For ``mode: external``, the configured URL is returned as-is.

    Args:
        config_tag: Key in the ``postgres`` YAML section.

    Returns:
        ``asyncpg``-compatible PostgreSQL connection string.
    """
    try:
        raw = global_config().get_dict(f"postgres.{config_tag}")
        config = PostgresConfig.model_validate(raw)
    except (ValueError, KeyError):
        logger.debug(
            "No 'postgres.{}' config found — falling back to 'vector_store.postgres_url'",
            config_tag,
        )
        try:
            return global_config().get_dsn("vector_store.postgres_url", "asyncpg")
        except Exception as exc:
            raise ValueError(
                f"No PostgreSQL configuration found for tag '{config_tag}'. "
                "Add a 'postgres' section to your YAML config."
            ) from exc

    if config.mode == "pgembed":
        return _start_pgembed(config)

    if not config.url:
        raise ValueError(f"PostgreSQL config '{config_tag}' has mode='external' but no 'url' specified.")
    return config.url


_pgembed_servers: dict[str, Any] = {}


def _start_pgembed(config: PostgresConfig) -> str:
    """Start an embedded PostgreSQL server via pgembed and return a connection URL.

    Args:
        config: PostgresConfig with ``mode='pgembed'``.

    Returns:
        asyncpg-compatible connection string for the embedded server.
    """
    try:
        import pgembed  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pgembed is required for embedded PostgreSQL. Install it with: uv add pgembed pgembed-pgvector"
        ) from exc

    data_dir = config.data_dir or "data/pgembed"

    if data_dir in _pgembed_servers:
        server = _pgembed_servers[data_dir]
    else:
        logger.info("Starting embedded PostgreSQL server at '{}'", data_dir)
        server = pgembed.get_server(data_dir)
        _pgembed_servers[data_dir] = server
        logger.info("pgembed server started on port {}", server.port)

    for ext in config.extensions:
        try:
            if pgembed.has_extension(ext):
                server.create_extension(ext)
                logger.debug("pgembed: installed extension '{}'", ext)
            else:
                logger.warning("pgembed: extension '{}' is not available on this platform", ext)
        except Exception as exc:
            logger.debug("pgembed extension '{}': {}", ext, exc)

    return f"postgresql+asyncpg://localhost:{server.port}/postgres"


# ---------------------------------------------------------------------------
# PgVector store factory
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
        postgres: Config tag in the ``postgres:`` YAML section.
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
        A ``(PGVectorStore, PGEngine)`` tuple.
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

    pg_engine: PGEngine = get_pg_engine(config.postgres)

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

    if config.hybrid_search and lc_hybrid_cfg is not None:
        _apply_hybrid_index_sync(vector_store)

    return vector_store, pg_engine


def _apply_hybrid_index_sync(vector_store: Any) -> None:
    """Apply the hybrid search GIN index, safely bridging sync/async."""

    async def _apply() -> None:
        await vector_store.apply_hybrid_search_index()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _apply())
            future.result(timeout=30)
        logger.info("Applied hybrid search GIN index on '{}'", vector_store.table_name)
    except Exception as exc:
        logger.warning("Could not apply hybrid search index (non-fatal): {}", exc)


# ---------------------------------------------------------------------------
# Backend class used by EmbeddingsStore
# ---------------------------------------------------------------------------


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
