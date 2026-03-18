"""Tests for chain registry functionality with fake models.

This module contains tests for the chain registry which manages
reusable AI processing chains using fake models.
"""

from unittest.mock import patch

from langchain_core.runnables import Runnable

from genai_tk.core.chain_registry import ChainRegistry, Example, RunnableItem, register_runnable
from genai_tk.core.llm_factory import get_llm

# Constants
FAKE_LLM_ID = "parrot_local@fake"


def test_registry_initialization() -> None:
    """Test that registry initializes correctly."""
    registry = ChainRegistry(registry=[])
    assert isinstance(registry.registry, list)
    assert len(registry.registry) == 0


def test_register_runnable_item() -> None:
    """Test registering a new RunnableItem."""
    registry = ChainRegistry(registry=[])

    # Create a simple chain
    llm = get_llm(llm=FAKE_LLM_ID)

    # Create a RunnableItem
    item = RunnableItem(name="test_chain", runnable=llm, examples=[Example(query=["test input"])])

    # Register the item
    registry.register(item)

    assert len(registry.registry) == 1
    assert registry.registry[0].name == "test_chain"


def test_find_runnable() -> None:
    """Test finding a registered runnable."""
    registry = ChainRegistry(registry=[])

    # Create and register a runnable
    llm = get_llm(llm=FAKE_LLM_ID)
    item = RunnableItem(name="test_chain", runnable=llm, examples=[Example(query=["test input"])])
    registry.register(item)

    # Find the runnable
    found_item = registry.find("test_chain")
    assert found_item is not None
    assert found_item.name == "test_chain"


def test_find_nonexistent_runnable() -> None:
    """Test finding a non-existent runnable."""
    registry = ChainRegistry(registry=[])

    found_item = registry.find("nonexistent_chain")
    assert found_item is None


def test_find_case_insensitive() -> None:
    """Test that find is case insensitive."""
    registry = ChainRegistry(registry=[])

    # Create and register a runnable
    llm = get_llm(llm=FAKE_LLM_ID)
    item = RunnableItem(name="Test_Chain", runnable=llm, examples=[Example(query=["test input"])])
    registry.register(item)

    # Should find with different cases
    assert registry.find("test_chain") is not None
    assert registry.find("TEST_CHAIN") is not None
    assert registry.find("Test_Chain") is not None


def test_get_runnable_list() -> None:
    """Test getting list of all runnables."""
    registry = ChainRegistry(registry=[])

    # Initially empty
    assert registry.get_runnable_list() == []

    # Register some runnables
    llm = get_llm(llm=FAKE_LLM_ID)
    item1 = RunnableItem(name="chain1", runnable=llm)
    item2 = RunnableItem(name="chain2", runnable=llm)

    registry.register(item1)
    registry.register(item2)

    runnables = registry.get_runnable_list()
    assert len(runnables) == 2
    assert runnables[0].name == "chain1"
    assert runnables[1].name == "chain2"


def test_runnable_item_creation() -> None:
    """Test RunnableItem creation and validation."""
    llm = get_llm(llm=FAKE_LLM_ID)

    # Test with runnable instance
    item = RunnableItem(name="test_item", runnable=llm, examples=[Example(query=["test"])])

    assert item.name == "test_item"
    assert item.runnable == llm
    assert len(item.examples) == 1
    assert item.examples[0].query == ["test"]


def test_runnable_item_execution() -> None:
    """Test executing a runnable from registry."""
    registry = ChainRegistry(registry=[])

    # Create and register a runnable
    llm = get_llm(llm=FAKE_LLM_ID)
    item = RunnableItem(name="exec_chain", runnable=llm, examples=[Example(query=["test input"])])
    registry.register(item)

    # Find and execute
    found_item = registry.find("exec_chain")
    assert found_item is not None

    # For simple runnable instances, test that we can access the runnable
    assert found_item.runnable is not None
    assert hasattr(found_item.runnable, "invoke")


def test_example_creation() -> None:
    """Test Example creation and validation."""
    # Test with query only
    example1 = Example(query=["test query"])
    assert example1.query == ["test query"]

    # Test with multiple queries
    example2 = Example(query=["query1", "query2"])
    assert example2.query == ["query1", "query2"]


def test_registry_with_multiple_similar_names() -> None:
    """Test registry behavior with similar names."""
    registry = ChainRegistry(registry=[])

    llm = get_llm(llm=FAKE_LLM_ID)

    # Register chains with similar names
    item1 = RunnableItem(name="chain", runnable=llm)
    item2 = RunnableItem(name="chain_v2", runnable=llm)
    item3 = RunnableItem(name="Chain", runnable=llm)  # Different case

    registry.register(item1)
    registry.register(item2)
    registry.register(item3)

    # Test finding with exact matches
    assert registry.find("chain") is not None
    assert registry.find("chain_v2") is not None
    assert registry.find("Chain") is not None

    # Test case insensitive search
    found = registry.find("CHAIN")
    # Should find one of them (first match)
    assert found is not None
    assert found.name.lower() == "chain"


def test_registry_performance(performance_threshold) -> None:
    """Test registry performance with many items."""
    import time

    registry = ChainRegistry(registry=[])
    llm = get_llm(llm=FAKE_LLM_ID)

    # Measure registration time
    start_time = time.time()
    for i in range(50):  # Reduced count for faster tests
        item = RunnableItem(name=f"chain_{i}", runnable=llm, examples=[Example(query=[f"test_{i}"])])
        registry.register(item)
    registration_time = time.time() - start_time

    # Measure search time
    start_time = time.time()
    for i in range(50):
        found = registry.find(f"chain_{i}")
        assert found is not None
    search_time = time.time() - start_time

    # Should be reasonably fast
    assert registration_time < performance_threshold * 2  # Allow more time for registration
    assert search_time < performance_threshold


# Factory and key-function tests require more complex setup
# and depend on internal implementation details


def test_runnable_item_get_direct_runnable() -> None:
    """Test RunnableItem.get() returns the Runnable instance directly."""
    llm = get_llm(llm=FAKE_LLM_ID)
    item = RunnableItem(name="test_direct", runnable=llm)
    result = item.get()
    assert result is llm
    assert isinstance(result, Runnable)


def test_runnable_item_get_callable_factory() -> None:
    """Test RunnableItem.get() invokes a callable factory with conf dict."""
    llm = get_llm(llm=FAKE_LLM_ID)

    def factory(conf: dict) -> Runnable:
        return llm

    item = RunnableItem(name="test_callable", runnable=factory)
    result = item.get({"llm": None})
    assert result is llm


def test_runnable_item_get_callable_default_conf() -> None:
    """Test RunnableItem.get() uses default conf {"llm": None} when conf=None."""
    llm = get_llm(llm=FAKE_LLM_ID)
    received_conf: dict = {}

    def factory(conf: dict) -> Runnable:
        received_conf.update(conf)
        return llm

    item = RunnableItem(name="test_callable_default", runnable=factory)
    result = item.get()  # conf=None triggers default
    assert result is llm
    assert received_conf == {"llm": None}


def test_runnable_item_get_tuple() -> None:
    """Test RunnableItem.get() handles (key, factory) tuple branch."""
    llm = get_llm(llm=FAKE_LLM_ID)

    def factory(conf: dict) -> Runnable:
        return llm

    item = RunnableItem(name="test_tuple", runnable=("input_key", factory))
    result = item.get({"llm": None})
    assert isinstance(result, Runnable)


def test_to_key_param_callable() -> None:
    """Test _to_key_param_callable creates a Runnable pipeline wrapping the key."""
    llm = get_llm(llm=FAKE_LLM_ID)
    factory = lambda conf: llm  # noqa: E731

    wrapped = ChainRegistry._to_key_param_callable("my_key", factory)
    assert callable(wrapped)
    result = wrapped({"llm": None})
    assert isinstance(result, Runnable)


def test_register_runnable_function() -> None:
    """Test module-level register_runnable() adds item to the global registry."""
    llm = get_llm(llm=FAKE_LLM_ID)
    unique_name = "test_register_via_module_function_unique"
    item = RunnableItem(name=unique_name, runnable=llm)

    register_runnable(item)

    registry = ChainRegistry.instance()
    found = registry.find(unique_name)
    assert found is not None
    assert found.name == unique_name


def test_load_modules_valid_module() -> None:
    """Test load_modules() successfully imports a valid module."""
    ChainRegistry.load_modules.invalidate()
    with patch("genai_tk.core.chain_registry.global_config") as mock_config:
        mock_config.return_value.get_list.return_value = ["genai_tk.core.chain_registry"]
        ChainRegistry.load_modules()  # Should not raise
    ChainRegistry.load_modules.invalidate()


def test_load_modules_invalid_module() -> None:
    """Test load_modules() gracefully handles an unimportable module."""
    ChainRegistry.load_modules.invalidate()
    with patch("genai_tk.core.chain_registry.global_config") as mock_config:
        mock_config.return_value.get_list.return_value = ["nonexistent.bogus.module.xyz"]
        ChainRegistry.load_modules()  # Should not raise, exception is logged
    ChainRegistry.load_modules.invalidate()


def test_runnable_item_get_invalid_type_raises() -> None:
    """Test RunnableItem.get() raises when runnable is an unsupported type."""
    import pytest

    # Use model_construct to bypass Pydantic validation and inject an invalid type
    item = RunnableItem.model_construct(name="bad", runnable=42)
    with pytest.raises(Exception, match="unknown or ill-formatted Runnable"):
        item.get()
