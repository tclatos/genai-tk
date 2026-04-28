"""Centralized PostgreSQL connection management.

Supports both external PostgreSQL servers and embedded PostgreSQL via pgembed,
providing a unified `get_pg_engine()` entry point for all vector store operations.

Configuration examples:
    ```yaml
    # config/baseline.yaml or project overrides
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
    from genai_tk.extra.postgres import get_pg_engine, get_postgres_url

    engine = get_pg_engine("default")  # returns PGEngine (singleton per tag)
    url = get_postgres_url("embedded")  # starts pgembed if needed
    ```
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


class PostgresConfig(BaseModel):
    """Configuration for a PostgreSQL connection source."""

    mode: str = "external"
    url: str | None = None
    data_dir: str | None = None
    extensions: list[str] = Field(default_factory=lambda: ["vector"])


# ---------------------------------------------------------------------------
# Engine cache (singleton per config tag)
# ---------------------------------------------------------------------------

_engine_cache: dict[str, Any] = {}


def get_pg_engine(config_tag: str = "default") -> Any:
    """Return a cached PGEngine for the given config tag.

    A new engine is created on the first call and reused on subsequent calls,
    which enables connection pooling without the caller managing the lifecycle.

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
        # Fall back to legacy vector_store.postgres_url
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


# ---------------------------------------------------------------------------
# pgembed support
# ---------------------------------------------------------------------------

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

    # Reuse running server for the same data_dir
    if data_dir in _pgembed_servers:
        server = _pgembed_servers[data_dir]
    else:
        logger.info("Starting embedded PostgreSQL server at '{}'", data_dir)
        server = pgembed.get_server(data_dir)
        _pgembed_servers[data_dir] = server
        logger.info("pgembed server started on port {}", server.port)

    # Install requested extensions
    for ext in config.extensions:
        try:
            if pgembed.has_extension(ext):
                server.create_extension(ext)
                logger.debug("pgembed: installed extension '{}'", ext)
            else:
                logger.warning("pgembed: extension '{}' is not available on this platform", ext)
        except Exception as exc:  # extension may already exist
            logger.debug("pgembed extension '{}': {}", ext, exc)

    # Build asyncpg-compatible URL
    return f"postgresql+asyncpg://localhost:{server.port}/postgres"
