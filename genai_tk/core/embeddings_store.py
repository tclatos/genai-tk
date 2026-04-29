"""Vector Store Factory.

Creates and manages LangChain VectorStore instances across multiple storage backends.
This module is intentionally scoped to *store creation only* — document ingestion,
retrieval, and deduplication belong to ``genai_tk.core.retriever_factory``.

Supported backends:
    - ``Chroma`` — persistent or in-memory
    - ``InMemory`` — ephemeral in-process store
    - ``PgVector`` — PostgreSQL with optional hybrid search

Configuration example:
    ```yaml
    embeddings_store:
      default:
        backend: Chroma
        embeddings: default
        config:
          storage: '::memory::'

      persistent:
        backend: Chroma
        embeddings: default
        table_name_prefix: my_docs
        config:
          storage: ${paths.data_root}/vector_store

      pg_store:
        backend: PgVector
        embeddings: default
        config:
          postgres: default          # references postgres: config tag
          hybrid_search: false
    ```

Usage:
    ```python
    es = EmbeddingsStore.create_from_config("default")
    vs = es.get_vector_store()  # returns LangChain VectorStore
    results = vs.similarity_search("query", k=4)
    ```
"""

from __future__ import annotations

import importlib
from typing import Annotated, Any, Literal, get_args

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# Backend literal type
# ---------------------------------------------------------------------------

VECTOR_STORE_ENGINE = Literal["Chroma", "InMemory", "PgVector"]


# ---------------------------------------------------------------------------
# Internal config model (parsed from YAML)
# ---------------------------------------------------------------------------


class _EmbeddingsStoreConfig(BaseModel):
    """Parsed configuration for an EmbeddingsStore YAML entry."""

    backend: str = Field(validation_alias=AliasChoices("backend", "id"))
    embeddings: str | None = None
    embeddings_id: str | None = None
    table_name_prefix: str = "embeddings"
    collection_metadata: dict[str, str] | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")  # silently ignore removed fields (record_manager etc.)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate legacy config keys to the current schema."""
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        # id → backend
        if "backend" not in migrated and "id" in migrated:
            migrated["backend"] = migrated["id"]
        # chroma_path → config.storage
        cfg = migrated.get("config")
        if isinstance(cfg, dict) and "storage" not in cfg and "chroma_path" in cfg:
            new_cfg = dict(cfg)
            new_cfg["storage"] = new_cfg.pop("chroma_path")
            migrated["config"] = new_cfg
        return migrated

    @model_validator(mode="after")
    def _validate(self) -> "_EmbeddingsStoreConfig":
        if self.embeddings and self.embeddings_id:
            raise ValueError("Cannot specify both 'embeddings_id' and 'embeddings' in embeddings_store config")
        return self

    def resolve_embeddings_factory(self) -> EmbeddingsFactory:
        """Resolve the appropriate EmbeddingsFactory from config fields."""
        if self.embeddings_id:
            return EmbeddingsFactory(embeddings=self.embeddings_id)
        if self.embeddings:
            return EmbeddingsFactory(embeddings=self.embeddings)
        return EmbeddingsFactory()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EmbeddingsStore(BaseModel):
    """Factory for creating LangChain VectorStore instances from YAML configuration.

    Must be created via ``create_from_config()``.  Direct instantiation is blocked.

    Example:
        ```python
        es = EmbeddingsStore.create_from_config("default")
        vs = es.get_vector_store()
        vs.add_documents(my_docs)
        results = vs.similarity_search("query", k=4)
        ```
    """

    backend: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True, alias="id")] = None
    embeddings_factory: EmbeddingsFactory
    table_name_prefix: str = "embeddings"
    config: dict[str, Any] = {}
    collection_metadata: dict[str, str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        raise RuntimeError(
            "EmbeddingsStore cannot be instantiated directly. "
            "Use EmbeddingsStore.create_from_config(config_tag) instead."
        )

    def model_post_init(self, __context: Any) -> None:
        if self.backend == "Chroma":
            storage = self.config.get("storage") or self.config.get("chroma_path")
            if not storage:
                self.config["storage"] = "::memory::"
            elif "chroma_path" in self.config and "storage" not in self.config:
                self.config["storage"] = self.config.pop("chroma_path")

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field
    @property
    def table_name(self) -> str:
        """Unique collection/table name derived from prefix and embeddings ID."""
        return f"{self.table_name_prefix}_{self.embeddings_factory.short_name()}"

    @computed_field
    @property
    def description(self) -> str:
        """Human-readable description of the store configuration."""
        r = f"{self.backend}/{self.table_name}"
        if self.backend == "Chroma":
            storage = self.config.get("storage", "::memory::")
            r += " => 'in-memory'" if storage == "::memory::" else " => 'on disk'"
        return r

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_from_config(cls, config_tag: str) -> "EmbeddingsStore":
        """Create an EmbeddingsStore from a named YAML configuration.

        Args:
            config_tag: Key in the ``embeddings_store`` YAML section.

        Returns:
            Configured EmbeddingsStore instance.
        """
        try:
            cfg = _EmbeddingsStoreConfig.model_validate(global_config().get_dict(f"embeddings_store.{config_tag}"))
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"embeddings_store configuration '{config_tag}' not found. Available: {cls.list_available_configs()}"
            ) from exc

        instance = cls.model_construct(
            backend=cfg.backend,
            embeddings_factory=cfg.resolve_embeddings_factory(),
            table_name_prefix=cfg.table_name_prefix,
            config=cfg.config,
            collection_metadata=cfg.collection_metadata,
        )
        instance.model_post_init(None)
        return instance

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("backend", mode="before")
    @classmethod
    def check_known(cls, backend: str | None) -> str:
        if backend is None:
            backend = global_config().get_str("vector_store.default")
        if backend == "Chroma_in_memory":
            logger.warning("Chroma_in_memory is deprecated — use backend='Chroma' with storage='::memory::'")
            backend = "Chroma"
        if backend == "Sklearn":
            raise ValueError("The Sklearn backend has been removed. Use 'Chroma' or 'InMemory' instead.")
        # Accept qualified class names and normalise to the short alias
        from genai_tk.core.vector_backends import ALIASES

        _qualified_to_short = {v: k for k, v in ALIASES.items()}
        if backend in _qualified_to_short:
            backend = _qualified_to_short[backend]
        if backend not in get_args(VECTOR_STORE_ENGINE):
            raise ValueError(
                f"Unknown vector store backend: '{backend}'. "
                f"Supported short names: {list(get_args(VECTOR_STORE_ENGINE))}. "
                f"Or use a qualified name like 'genai_tk.core.vector_backends.ChromaBackend'."
            )
        return backend

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def get_vector_store(self) -> VectorStore:
        """Create and return a configured LangChain VectorStore.

        Returns:
            Configured vector store instance ready for add_documents / similarity_search.
        """
        from genai_tk.core.vector_backends import ALIASES

        embeddings = self.embeddings_factory.get()
        qualified = ALIASES.get(self.backend or "", self.backend or "")
        module_path, _, class_name = qualified.rpartition(".")
        backend_cls = getattr(importlib.import_module(module_path), class_name)

        if self.backend == "PgVector":
            vector_store = backend_cls.create_from_factory(
                embeddings_factory=self.embeddings_factory,
                table_name=self.table_name,
                config=self.config,
                collection_metadata=self.collection_metadata,
            )
        else:
            vector_store = backend_cls.create(
                embeddings=embeddings,
                table_name=self.table_name,
                config=self.config,
                collection_metadata=self.collection_metadata,
            )

        logger.debug("Created vector store: {}", self.description)
        return vector_store

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def document_count(self) -> int:
        """Return the number of documents in the store.

        Only supported for Chroma backends.
        """
        if self.backend == "Chroma":
            return self.get_vector_store()._collection.count()  # type: ignore[attr-defined]
        raise NotImplementedError(f"document_count() not supported for backend '{self.backend}'")

    def get_stats(self) -> dict[str, Any]:
        """Return a diagnostic dictionary describing the store."""
        stats: dict[str, Any] = {
            "backend": self.backend,
            "table_name": self.table_name,
            "description": self.description,
            "embeddings_model": self.embeddings_factory.embeddings_id,
        }
        try:
            stats["document_count"] = self.document_count()
        except (NotImplementedError, Exception):
            stats["document_count"] = "unknown"

        if self.backend == "Chroma":
            storage = self.config.get("storage", "::memory::")
            stats["storage_type"] = "memory" if storage == "::memory::" else "persistent"
            if storage != "::memory::":
                stats["storage_path"] = storage
        return stats

    @staticmethod
    def known_items() -> list[str]:
        """List supported vector store backend identifiers."""
        return list(get_args(VECTOR_STORE_ENGINE))

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List all config tags available in the ``embeddings_store`` YAML section."""
        try:
            cfg = global_config().get("embeddings_store", {})
            return list(cfg.keys()) if hasattr(cfg, "keys") else []
        except Exception:
            return []

    # Backend-specific helpers kept for existing callers that may import them directly.
    # New code should use get_vector_store() which dispatches via the backend class.

    def _create_chroma_vector_store(self, embeddings: Embeddings) -> VectorStore:
        from genai_tk.core.vector_backends.chroma import ChromaBackend

        return ChromaBackend.create(
            embeddings=embeddings,
            table_name=self.table_name,
            config=self.config,
            collection_metadata=self.collection_metadata,
        )

    def _create_pg_vector_store(self) -> VectorStore:
        from genai_tk.core.vector_backends.pgvector import PgVectorBackend

        return PgVectorBackend.create_from_factory(
            embeddings_factory=self.embeddings_factory,
            table_name=self.table_name,
            config=self.config,
            collection_metadata=self.collection_metadata,
        )


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------


def search_one(vc: VectorStore, query: str) -> list:
    """Return the single most similar document from a vector store."""
    return vc.similarity_search(query, k=1)
