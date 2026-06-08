"""Async key-value storage for Pydantic objects using py-key-value-aio.

Provides PydanticStore, an async helper that persists and retrieves any
Pydantic BaseModel via a configured AsyncKeyValue backend.  The store is
selected by name from the ``kv_store`` config section; the Pydantic class name
is used as the collection (namespace) within the store.
"""

from __future__ import annotations

import json
from typing import TypeVar

from key_value.aio.adapters.base_model import BaseModelAdapter
from loguru import logger
from pydantic import BaseModel

from genai_tk.extra.kv_store_factory import get_async_kv_store
from genai_tk.utils.hashing import buffer_digest

T = TypeVar("T", bound=BaseModel)


class PydanticStore(BaseModel):
    """Async store for persisting and retrieving Pydantic objects.

    Args:
        kvstore_id: Key in the ``kv_store`` config section (e.g. ``"default"``).
        model: Pydantic model class to store/retrieve.
    """

    kvstore_id: str
    model: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}

    async def save_obj(self, key: str | dict, obj: BaseModel) -> None:
        """Persist a Pydantic model to the key-value store.

        Args:
            key: Unique identifier (str or dict; dict keys are MD5-hashed).
            obj: Pydantic model instance to persist.
        """
        str_key = _make_key(key)
        store = get_async_kv_store(self.kvstore_id)
        adapter = BaseModelAdapter(
            key_value=store,
            pydantic_model=self.model,
            default_collection=self.model.__name__,
        )
        await adapter.put(key=str_key, value=obj)
        logger.debug("saved '{}/{}' to kv_store '{}'", self.model.__name__, str_key, self.kvstore_id)

    async def load_object(self, key: str | dict) -> T | None:
        """Load a Pydantic model from the key-value store.

        Args:
            key: Unique identifier for the stored object.

        Returns:
            Validated model instance, or None if the key is not found.
        """
        str_key = _make_key(key)
        store = get_async_kv_store(self.kvstore_id)
        adapter = BaseModelAdapter(
            key_value=store,
            pydantic_model=self.model,
            default_collection=self.model.__name__,
        )
        result = await adapter.get(key=str_key)
        if result is not None:
            logger.debug("read '{}/{}' from kv_store '{}'", self.model.__name__, str_key, self.kvstore_id)
        return result  # type: ignore[return-value]


def _make_key(key: str | dict) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, dict):
        return buffer_digest(json.dumps(key, sort_keys=True).encode())
    raise ValueError("key must be str or dict")
