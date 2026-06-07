"""Factory for creating and managing key-value stores with LangChain storage backends.

Provides a factory pattern for creating ByteStore instances with support for
different storage backends like local file storage and PostgreSQL.

Supports configuration-based store creation with multiple named configurations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Literal

from langchain_classic.storage import LocalFileStore
from langchain_core.stores import ByteStore
from pydantic import BaseModel, Discriminator, Tag

from genai_tk.utils.config_mngr import global_config

# ---------------------------------------------------------------------------
# Discriminated-union config models
# ---------------------------------------------------------------------------


class LocalFileStoreConfig(BaseModel):
    """Configuration for local file-based key-value store."""

    type: Literal["LocalFileStore"] = "LocalFileStore"
    path: str | Path

    def create_store(self, namespace: str = "") -> ByteStore:
        base_path = Path(self.path)
        store_path = base_path / namespace if namespace else base_path
        store_path.mkdir(parents=True, exist_ok=True)
        return LocalFileStore(str(store_path))


class SQLStoreConfig(BaseModel):
    """Configuration for SQL-based key-value store."""

    type: Literal["SQLStore"] = "SQLStore"
    path: str

    def create_store(self, namespace: str = "") -> ByteStore:
        from langchain_community.storage import SQLStore

        store = SQLStore(namespace=namespace, db_url=self.path)
        store.create_schema()
        return store


class MemoryStoreConfig(BaseModel):
    """Configuration for ephemeral in-memory key-value store (tests)."""

    type: Literal["memory"] = "memory"

    def create_store(self, namespace: str = "") -> ByteStore:
        temp_dir = tempfile.mkdtemp()
        return LocalFileStore(str(Path(temp_dir) / namespace) if namespace else temp_dir)


def _kv_store_discriminator(v: dict | BaseModel) -> str:
    if isinstance(v, dict):
        return v.get("type", "LocalFileStore")
    return getattr(v, "type", "LocalFileStore")


KvStoreConfig = Annotated[
    Annotated[LocalFileStoreConfig, Tag("LocalFileStore")]
    | Annotated[SQLStoreConfig, Tag("SQLStore")]
    | Annotated[MemoryStoreConfig, Tag("memory")],
    Discriminator(_kv_store_discriminator),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KvStoreRegistry(BaseModel):
    """Registry for creating key-value stores with configurable backends."""

    def _get_store_config(self, store_id: str) -> LocalFileStoreConfig | SQLStoreConfig | MemoryStoreConfig:
        stores = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)  # type: ignore[arg-type]
        if store_id not in stores:
            available = list(stores.keys())
            raise ValueError(f"Store configuration '{store_id}' not found. Available: {available}")
        return stores[store_id]

    def get(self, store_id: str = "default", namespace: str = "") -> ByteStore:
        """Create and return a ByteStore instance based on configuration."""
        config = self._get_store_config(store_id)
        return config.create_store(namespace)

    @staticmethod
    def get_available_stores() -> list[str]:
        stores = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)  # type: ignore[arg-type]
        return list(stores.keys())


def get_kv_store(store_id: str = "default", namespace: str = "") -> ByteStore:
    """Create a configured ByteStore instance."""
    registry = KvStoreRegistry()
    return registry.get(store_id=store_id, namespace=namespace)
