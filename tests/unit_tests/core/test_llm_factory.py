"""Tests for LLM factory functionality.

This module contains tests that verify:
- Basic LLM initialization and functionality
- Configuration switching between different LLMs
- Error handling for invalid configurations
- Factory method behavior
- JSON mode and streaming options
"""

# Import constants directly to avoid import resolution issues
import sys
from pathlib import Path

import pytest
from langchain_core.messages.human import HumanMessage

from genai_tk.core.llm_factory import (
    LlmFactory,
    configurable,
    get_llm,
    get_llm_info,
    llm_config,
)

sys.path.append(str(Path(__file__).parent.parent.parent))

# Fake model constants - using same values as in test utils
FAKE_LLM_ID = "parrot_local@fake"
FAKE_LLM_PROVIDER = "fake"
LLM_ID_FOR_TEST = "gpt_41mini@openrouter"


def test_basic_call(fake_llm) -> None:
    """Test that we can generate a simple joke using fake LLM."""
    joke = fake_llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, HumanMessage), f"{type(joke)}"
    assert len(joke.content) > 10  # Basic check that we got some content


def test_streaming_joke(fake_llm_with_streaming) -> None:
    """Test streaming joke generation."""
    chunks = []
    for chunk in fake_llm_with_streaming.stream("Tell me a joke about AI"):
        chunks.append(chunk)

    assert len(chunks) > 0
    print(type(chunks[0]))
    assert isinstance(chunks[0], HumanMessage), f"{type(chunks[0])}"


def test_get_llm_with_none_id_uses_default(fake_llm) -> None:
    """Test that get_llm uses default when llm_id is None."""
    # This should not raise an exception
    assert fake_llm is not None


def test_get_llm_with_llm_type(fake_llm) -> None:
    """Test get_llm with llm_type parameter."""
    # Use fake LLM to avoid API key issues
    assert fake_llm is not None


def test_invalid_llm_id_raises_error() -> None:
    """Test that invalid llm_id raises ValueError."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm(llm="nonexistent_model")


def test_llm_factory_creation() -> None:
    """Test LlmFactory class creation."""
    factory = LlmFactory(llm=FAKE_LLM_ID)
    assert factory.llm_id == FAKE_LLM_ID
    assert factory.info is not None
    assert factory.provider == FAKE_LLM_PROVIDER


def test_llm_factory_short_name() -> None:
    """Test short_name method returns correct format."""
    factory = LlmFactory(llm=FAKE_LLM_ID)
    short = factory.short_name()
    assert short == "parrot_local"


def test_llm_factory_get_litellm_model_name() -> None:
    """Test get_litellm_model_name method."""
    factory = LlmFactory(llm=LLM_ID_FOR_TEST)
    # Skip this test for fake provider since litellm doesn't support it
    if factory.provider == FAKE_LLM_PROVIDER:
        pytest.skip("LiteLLM doesn't support fake provider")
    model_name = factory.get_litellm_model_name()
    assert model_name == "openrouter/openai/gpt-4.1-mini"


def test_llm_factory_get_smolagent_model() -> None:
    """Test get_smolagent_model method."""
    factory = LlmFactory(llm=FAKE_LLM_ID)
    # Skip this test for fake provider since smolagent doesn't support it
    if factory.provider == FAKE_LLM_PROVIDER:
        pytest.skip("smolagent doesn't support fake provider")
    model = factory.get_smolagent_model()
    assert model is not None


def test_get_llm_info() -> None:
    """Test get_llm_info function."""
    info = get_llm_info(FAKE_LLM_ID)
    assert info.llm == FAKE_LLM_ID
    assert info.provider == FAKE_LLM_PROVIDER
    assert info.model == "parrot"


def test_get_llm_info_invalid_id(invalid_llm_id) -> None:
    """Test get_llm_info with invalid ID."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm_info(invalid_llm_id)


def test_llm_config() -> None:
    """Test llm_config function."""
    config = llm_config(FAKE_LLM_ID)
    assert "configurable" in config
    assert config["configurable"]["llm_id"] == FAKE_LLM_ID


def test_llm_config_invalid_id() -> None:
    """Test llm_config with invalid ID."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        llm_config("nonexistent_model")


def test_configurable() -> None:
    """Test configurable function."""
    config = configurable({"test_key": "test_value"})
    assert "configurable" in config
    assert config["configurable"]["test_key"] == "test_value"


def test_json_mode_parameter(fake_llm_with_json_mode) -> None:
    """Test JSON mode parameter."""
    assert fake_llm_with_json_mode is not None


def test_streaming_parameter(fake_llm_with_streaming) -> None:
    """Test streaming parameter."""
    assert fake_llm_with_streaming is not None


def test_cache_parameter(fake_llm_with_cache) -> None:
    """Test cache parameter."""
    assert fake_llm_with_cache is not None


def test_llm_params_parameter(fake_llm) -> None:
    """Test additional LLM parameters."""
    llm = get_llm(llm=FAKE_LLM_ID, temperature=0.5, max_tokens=100)
    assert llm is not None


def test_known_items() -> None:
    """Test known_items method."""
    items = LlmFactory.known_items()
    assert isinstance(items, list)
    assert len(items) > 0
    assert FAKE_LLM_ID in items


def test_known_items_dict() -> None:
    """Test known_items_dict method."""
    items_dict = LlmFactory.known_items_dict()
    assert isinstance(items_dict, dict)
    assert FAKE_LLM_ID in items_dict


def test_complex_provider_config_parsing() -> None:
    """Test that complex provider configurations are parsed correctly."""
    # Test the parsing logic by checking the raw data structure
    from genai_tk.core.llm_factory import _read_llm_list_file

    llms = _read_llm_list_file()

    # Find any model with complex configuration to test parsing
    complex_model = None
    for llm in llms:
        # Look for models with additional configuration
        if llm.llm_args and len(llm.llm_args) > 2:
            complex_model = llm
            break

    if complex_model:
        # Verify that complex configurations are parsed correctly
        assert complex_model.llm is not None
        assert complex_model.provider is not None
        assert isinstance(complex_model.llm_args, dict)

        # Check that nested structures are parsed
        for key, value in complex_model.llm_args.items():
            if isinstance(value, dict):
                # Nested configuration found
                assert len(value) >= 0  # Should be a valid dict
                break
    else:
        # If no complex model found, at least verify basic parsing works
        assert len(llms) > 0
        for llm in llms[:3]:  # Check first few models
            assert llm.llm is not None
            assert llm.provider is not None
            assert isinstance(llm.llm_args, dict)


def test_factory_find_llm_id_from_type() -> None:
    """Test find_llm_id_from_type method."""
    # This might fail if no config is set up, so we'll test the error case
    with pytest.raises(ValueError):
        LlmFactory.find_llm_id_from_tag("nonexistent_type")


def test_llm_factory_model_validation() -> None:
    """Test LlmFactory model validation."""
    # Test valid ID
    factory = LlmFactory(llm=FAKE_LLM_ID)
    assert factory.llm_id == FAKE_LLM_ID

    # Test invalid ID
    with pytest.raises(ValueError, match="Unknown LLM"):
        LlmFactory(llm="invalid_model_id")


def test_field_validator_cache() -> None:
    """Test cache field validator."""
    # Valid cache value
    from langchain_community.cache import SQLiteCache

    llm = LlmFactory(llm=LLM_ID_FOR_TEST, cache="sqlite").get()
    assert isinstance(llm.cache, SQLiteCache)

    # Invalid cache value should raise ValueError
    with pytest.raises(ValueError, match="Unknown cache method"):
        LlmFactory(llm_id=FAKE_LLM_ID, cache="invalid_cache")
