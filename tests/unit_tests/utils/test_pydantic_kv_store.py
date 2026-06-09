"""Unit tests for kv_store module."""

import asyncio
import unittest

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


class TestMakeKey(unittest.TestCase):
    """Tests for the _make_key helper."""

    def test_string_key_unchanged(self):
        self.assertEqual(_make_key("hello"), "hello")

    def test_string_key_with_special_chars(self):
        # Raw keys are passed through as-is
        self.assertEqual(_make_key("some/path/file.json"), "some/path/file.json")

    def test_dict_key_is_hex(self):
        key = _make_key({"user_id": 123})
        # buffer_digest uses xxh3_64 by default (16 hex chars), not MD5 (32)
        self.assertRegex(key, r"^[0-9a-f]+$")

    def test_dict_key_deterministic(self):
        key1 = _make_key({"a": 1, "b": 2})
        key2 = _make_key({"b": 2, "a": 1})
        self.assertEqual(key1, key2)

    def test_invalid_key_type_raises(self):
        with self.assertRaises(ValueError):
            _make_key(123)  # type: ignore[arg-type]


class TestPydanticStore(unittest.TestCase):
    """Test PydanticStore async operations backed by MemoryStore."""

    def setUp(self):
        _configure_memory_store("test")
        self.test_model = SampleModel(name="test_object", value=42)
        self.store = PydanticStore(kvstore_id="test", model=SampleModel)

    def test_save_and_load(self):
        async def _run():
            await self.store.save_obj("key1", self.test_model)
            result = await self.store.load_object("key1")
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "test_object")
            self.assertEqual(result.value, 42)

        asyncio.run(_run())

    def test_load_missing_key_returns_none(self):
        async def _run():
            result = await self.store.load_object("nonexistent")
            self.assertIsNone(result)

        asyncio.run(_run())

    def test_dict_key(self):
        async def _run():
            dict_key = {"user_id": 123, "session": "abc"}
            await self.store.save_obj(dict_key, self.test_model)
            result = await self.store.load_object(dict_key)
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "test_object")

        asyncio.run(_run())

    def test_overwrite(self):
        async def _run():
            await self.store.save_obj("key1", self.test_model)
            new_model = SampleModel(name="updated", value=99)
            await self.store.save_obj("key1", new_model)
            result = await self.store.load_object("key1")
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "updated")
            self.assertEqual(result.value, 99)

        asyncio.run(_run())

    def test_multiple_model_types(self):
        class AnotherModel(BaseModel):
            title: str
            count: float

        _configure_memory_store("test2")

        async def _run():
            store2 = PydanticStore(kvstore_id="test2", model=AnotherModel)
            await self.store.save_obj("k1", SampleModel(name="m1", value=1))
            await store2.save_obj("k2", AnotherModel(title="m2", count=2.5))
            r1 = await self.store.load_object("k1")
            r2 = await store2.load_object("k2")
            self.assertEqual(r1.name, "m1")
            self.assertEqual(r2.title, "m2")

        asyncio.run(_run())

    def test_invalid_store_id_raises(self):
        async def _run():
            bad_store = PydanticStore(kvstore_id="nonexistent_store", model=SampleModel)
            with self.assertRaises(ValueError):
                await bad_store.save_obj("key", self.test_model)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
