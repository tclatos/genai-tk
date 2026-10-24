"""Preserve JSON-backed cache behavior without the unused DiskCache backend."""

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.extra.kv_store_factory import AsyncKeyValueByteStoreAdapter, KvStoreConfig, clear_store_cache
from genai_tk.utils.pydantic_utils.kv_store import PydanticStore

pytestmark = pytest.mark.integration


class CachedResult(BaseModel):
    """A structured result persisted through the public Pydantic store API."""

    text: str
    count: int


@pytest.fixture
def filetree_config(tmp_path: Path) -> KvStoreConfig:
    return KvStoreConfig(
        type="key_value.aio.stores.filetree.FileTreeStore",
        args={"data_directory": str(tmp_path / "cache")},
    )


@pytest.fixture
def configured_filetree(filetree_config: KvStoreConfig, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    config = global_config()
    store_id = "filetree_security_test"
    overrides = deepcopy(config.root.get(config.active_context, {}))
    monkeypatch.setitem(config.root, config.active_context, overrides)
    config.set(f"kv_store.{store_id}", filetree_config.model_dump())
    clear_store_cache()
    try:
        yield store_id
    finally:
        clear_store_cache()


async def test_filetree_persists_values_across_instances(filetree_config: KvStoreConfig) -> None:
    value = {"text": "cached result", "count": 2}
    await filetree_config.create_store().put(key="result", value=value, collection="results")

    reopened = filetree_config.create_store()
    assert await reopened.get(key="result", collection="results") == value


async def test_byte_store_persists_binary_values(filetree_config: KvStoreConfig) -> None:
    value = b"\x00\xff\x80binary\n"
    store = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="embeddings")
    await store.amset([("same_key", value)])

    reopened = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="embeddings")
    assert await reopened.amget(["same_key"]) == [value]


async def test_byte_store_isolates_collection_namespaces(filetree_config: KvStoreConfig) -> None:
    first = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="first")
    second = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="second")
    await first.amset([("same_key", b"private to first")])

    assert await second.amget(["same_key"]) == [None]


async def test_byte_store_deletion_persists(filetree_config: KvStoreConfig) -> None:
    store = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="embeddings")
    await store.amset([("result", b"cached")])
    await store.amdelete(["result"])

    reopened = AsyncKeyValueByteStoreAdapter(filetree_config.create_store(), collection="embeddings")
    assert await reopened.amget(["result"]) == [None]


async def test_pydantic_store_persists_models(configured_filetree: str) -> None:
    model = CachedResult(text="structured result", count=3)
    store = PydanticStore(kvstore_id=configured_filetree, model=CachedResult)
    await store.save_obj("result", model)
    clear_store_cache()

    reopened = PydanticStore(kvstore_id=configured_filetree, model=CachedResult)
    assert await reopened.load_object("result") == model
