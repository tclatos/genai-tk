"""Tests for chain registry functionality with fake models.

This module contains tests for the chain registry which manages
reusable AI processing chains using fake models.
"""

import pytest
from langchain_core.runnables import Runnable

from genai_tk.core.chain_registry import ChainRegistry, Example, RunnableItem
from genai_tk.core.llm_factory import get_llm

# Constants
FAKE_LLM_ID = "parrot_local_fake"


def test_registry_initialization() -> None:
    """Test that registry initializes correctly."""
    registry = ChainRegistry(registry=[])
    assert isinstance(registry.registry, list)
    assert len(registry.registry) == 0


def test_register_runnable_item() -> None:
    """Test registering a new RunnableItem."""
    registry = ChainRegistry(registry=[])

    # Create a simple chain
    llm = get_llm(llm_id=FAKE_LLM_ID)

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
    llm = get_llm(llm_id=FAKE_LLM_ID)
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
    llm = get_llm(llm_id=FAKE_LLM_ID)
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
    llm = get_llm(llm_id=FAKE_LLM_ID)
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
    llm = get_llm(llm_id=FAKE_LLM_ID)

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
    llm = get_llm(llm_id=FAKE_LLM_ID)
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

    llm = get_llm(llm_id=FAKE_LLM_ID)

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
    llm = get_llm(llm_id=FAKE_LLM_ID)

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
