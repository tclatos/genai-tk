"""Async key-value store factory using py-key-value-aio.

Provides a YAML-driven factory that instantiates any AsyncKeyValue-compatible
store via a dotted class path, and a LangChain AsyncByteStore adapter for use
with CacheBackedEmbeddings.

YAML configuration schema::

    kv_store:
      default:
        type: key_value.aio.stores.filetree.FileTreeStore
        args:
          directory: ${paths.data_root}/kv_store
      memory:
        type: key_value.aio.stores.memory.MemoryStore
        args: {}
      postgres:
        type: key_value.aio.stores.postgresql.PostgreSQLStore
        args:
          url: postgresql://user:pass@localhost:5432/mydb
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import importlib
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from key_value.aio.protocols.key_value import AsyncKeyValue
from langchain_core.stores import BaseStore
from pydantic import BaseModel

from genai_tk.utils.config_mngr import global_config


class KvStoreConfig(BaseModel):
    """Configuration for a single py-key-value-aio store entry."""

    type: str  # Dotted class path, e.g. "key_value.aio.stores.filetree.FileTreeStore"
    args: dict[str, Any] = {}

    def create_store(self) -> AsyncKeyValue:
        """Instantiate the store from its class path and args."""
        module_path, class_name = self.type.rsplit(".", 1)
        module = importlib.import_module(module_path)
        store_class = getattr(module, class_name)
        return store_class(**self.args)


# Module-level cache: store_id → AsyncKeyValue instance (shared across calls)
_store_cache: dict[str, AsyncKeyValue] = {}


def get_async_kv_store(store_id: str = "default") -> AsyncKeyValue:
    """Create and return an AsyncKeyValue instance from configuration.

    Instances are cached by store_id so the same backend is reused across calls.
    Call :func:`clear_store_cache` to force re-creation (e.g., in tests).

    Args:
        store_id: Key in the ``kv_store`` config section (e.g. ``"default"``).

    Returns:
        Configured AsyncKeyValue store instance.

    Raises:
        ValueError: If the store_id is not present in configuration.
    """
    if store_id not in _store_cache:
        stores = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)
        if store_id not in stores:
            available = list(stores.keys())
            raise ValueError(f"KV store '{store_id}' not found. Available: {available}")
        _store_cache[store_id] = stores[store_id].create_store()
    return _store_cache[store_id]


def clear_store_cache() -> None:
    """Clear the cached store instances.

    Call this in tests after reconfiguring ``kv_store`` entries, or any time
    you need fresh store instances to pick up config changes.
    """
    _store_cache.clear()


def get_available_stores() -> list[str]:
    """Return the list of configured KV store IDs."""
    stores = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)
    return list(stores.keys())


# ---------------------------------------------------------------------------
# LangChain bridge
# ---------------------------------------------------------------------------


class AsyncKeyValueByteStoreAdapter(BaseStore[str, bytes]):
    """Adapts an AsyncKeyValue store to the LangChain ByteStore protocol.

    Bytes are stored as base64-encoded strings inside a JSON dict so they fit
    the ``dict[str, Any]`` value type required by py-key-value-aio stores.
    Sync methods delegate to the async methods via ``asyncio.run()``, which is
    safe in sync-only contexts.  Async overrides use the native API directly.

    Args:
        kv_store: The underlying AsyncKeyValue instance.
        collection: Collection (namespace) to use within the store.
    """

    def __init__(self, kv_store: AsyncKeyValue, collection: str = "bytes") -> None:
        self._kv = kv_store
        self._collection = collection

    def _run(self, coro) -> Any:  # type: ignore[type-arg]
        """Run an async coroutine synchronously, safe in both contexts."""
        try:
            asyncio.get_running_loop()
            # Already inside an event loop — delegate to a worker thread.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    # ── Sync (required by BaseStore ABC) ─────────────────────────────────────

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        return self._run(self.amget(list(keys)))

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        self._run(self.amset(list(key_value_pairs)))

    def mdelete(self, keys: Sequence[str]) -> None:
        self._run(self.amdelete(list(keys)))

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        return iter([])

    # ── Async (native, preferred by LangChain async chains) ──────────────────

    async def amget(self, keys: Sequence[str]) -> list[bytes | None]:
        results = []
        for key in keys:
            value = await self._kv.get(key=key, collection=self._collection)
            results.append(base64.b64decode(value["data"]) if value is not None else None)
        return results

    async def amset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        for key, value in key_value_pairs:
            await self._kv.put(
                key=key,
                value={"data": base64.b64encode(value).decode()},
                collection=self._collection,
            )

    async def amdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            await self._kv.delete(key=key, collection=self._collection)

    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[str]:  # type: ignore[override]
        # Not supported — yields nothing.
        if False:  # noqa: SIM210
            yield ""
