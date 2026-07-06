"""Unit tests for the pydantic_utils.kv_store module (pytest style)."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from genai_tk.utils.pydantic_utils.kv_store import PydanticStore, _make_key


class SampleModel(BaseModel):
    """Test model for kv_store testing."""

    name: str
    value: int


def _configure_memory_store(store_id: str = "test") -> None:
    """Set up an in-memory store in global config for the given ID and reset cache."""
    from genai_tk.config_mgmt.config_mngr import global_config
    from genai_tk.extra.kv_store_factory import clear_store_cache

    global_config().set(f"kv_store.{store_id}.type", "key_value.aio.stores.memory.MemoryStore")
    global_config().set(f"kv_store.{store_id}.args", {})
    clear_store_cache()


@pytest.fixture
def store() -> PydanticStore:
    """Configure a fresh in-memory store and return a PydanticStore for SampleModel."""
    _configure_memory_store("test")
    return PydanticStore(kvstore_id="test", model=SampleModel)


@pytest.fixture
def test_model() -> SampleModel:
    """Return a sample model instance used across tests."""
    return SampleModel(name="test_object", value=42)


def test_string_key_unchanged() -> None:
    assert _make_key("hello") == "hello"


def test_string_key_with_special_chars() -> None:
    assert _make_key("some/path/file.json") == "some/path/file.json"


def test_dict_key_is_hex() -> None:
    key = _make_key({"user_id": 123})
    assert re.fullmatch(r"[0-9a-f]+", key)


def test_dict_key_deterministic() -> None:
    assert _make_key({"a": 1, "b": 2}) == _make_key({"b": 2, "a": 1})


def test_invalid_key_type_raises() -> None:
    with pytest.raises(ValueError):
        _make_key(123)  # type: ignore[arg-type]


async def test_save_and_load(store: PydanticStore, test_model: SampleModel) -> None:
    await store.save_obj("key1", test_model)
    result = await store.load_object("key1")
    assert result is not None
    assert result.name == "test_object"
    assert result.value == 42


async def test_load_missing_key_returns_none(store: PydanticStore) -> None:
    assert await store.load_object("nonexistent") is None


async def test_dict_key(store: PydanticStore, test_model: SampleModel) -> None:
    dict_key = {"user_id": 123, "session": "abc"}
    await store.save_obj(dict_key, test_model)
    result = await store.load_object(dict_key)
    assert result is not None
    assert result.name == "test_object"


async def test_overwrite(store: PydanticStore, test_model: SampleModel) -> None:
    await store.save_obj("key1", test_model)
    await store.save_obj("key1", SampleModel(name="updated", value=99))
    result = await store.load_object("key1")
    assert result is not None
    assert result.name == "updated"
    assert result.value == 99


async def test_multiple_model_types(store: PydanticStore) -> None:
    class AnotherModel(BaseModel):
        title: str
        count: float

    _configure_memory_store("test2")
    store2 = PydanticStore(kvstore_id="test2", model=AnotherModel)

    await store.save_obj("k1", SampleModel(name="m1", value=1))
    await store2.save_obj("k2", AnotherModel(title="m2", count=2.5))

    r1 = await store.load_object("k1")
    r2 = await store2.load_object("k2")
    assert r1.name == "m1"
    assert r2.title == "m2"


async def test_invalid_store_id_raises(test_model: SampleModel) -> None:
    bad_store = PydanticStore(kvstore_id="nonexistent_store", model=SampleModel)
    with pytest.raises(ValueError):
        await bad_store.save_obj("key", test_model)
