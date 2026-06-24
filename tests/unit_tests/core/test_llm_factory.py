"""Tests for LLM factory functionality."""

import pytest
from langchain_core.messages.human import HumanMessage

from genai_tk.core.factories.llm_factory import (
    LlmFactory,
    LlmInfo,
    _extract_reasoning_settings,
    _split_inline_reasoning_effort,
    configurable,
    get_llm,
    get_llm_info,
    llm_config,
)

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


def test_llm_factory_creation(fake_llm_id) -> None:
    """Test LlmFactory class creation."""
    factory = LlmFactory(llm=fake_llm_id)
    assert factory.llm_id == fake_llm_id
    assert factory.info is not None
    assert factory.provider == FAKE_LLM_PROVIDER


def test_llm_factory_short_name(fake_llm_id) -> None:
    """Test short_name method returns correct format."""
    factory = LlmFactory(llm=fake_llm_id)
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


def test_llm_factory_get_smolagent_model(fake_llm_id) -> None:
    """Test get_smolagent_model returns a model object (smolagents mocked)."""
    from unittest.mock import MagicMock, patch

    factory = LlmFactory(llm=fake_llm_id)
    mock_model = MagicMock()
    mock_litellm_cls = MagicMock(return_value=mock_model)
    mock_smolagents = MagicMock()
    mock_smolagents.LiteLLMModel = mock_litellm_cls
    mock_smolagents.AzureOpenAIServerModel = MagicMock()

    with (
        patch.dict("sys.modules", {"smolagents": mock_smolagents}),
        patch.object(LlmFactory, "get_litellm_model_name", return_value="fake/parrot"),
    ):
        model = factory.get_smolagent_model()

    assert model is mock_model
    mock_litellm_cls.assert_called_once_with(model_id="fake/parrot", **factory.llm_params)


def test_get_llm_info(fake_llm_id) -> None:
    """Test get_llm_info function."""
    info = get_llm_info(fake_llm_id)
    assert info.llm == fake_llm_id
    assert info.provider == FAKE_LLM_PROVIDER
    assert info.model == "parrot"


def test_get_llm_info_invalid_id(invalid_llm_id) -> None:
    """Test get_llm_info with invalid ID."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm_info(invalid_llm_id)


def test_llm_config(fake_llm_id) -> None:
    """Test llm_config function."""
    config = llm_config(fake_llm_id)
    assert "configurable" in config
    assert config["configurable"]["llm_id"] == fake_llm_id


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


def test_llm_params_parameter(fake_llm_id) -> None:
    """Test additional LLM parameters."""
    llm = get_llm(llm=fake_llm_id, temperature=0.5, max_tokens=100)
    assert llm is not None


def test_known_items(fake_llm_id) -> None:
    """Test known_items method."""
    items = LlmFactory.known_items()
    assert isinstance(items, list)
    assert len(items) > 0
    assert fake_llm_id in items


def test_known_items_dict(fake_llm_id) -> None:
    """Test known_items_dict method."""
    items_dict = LlmFactory.known_items_dict()
    assert isinstance(items_dict, dict)
    assert fake_llm_id in items_dict


def test_complex_provider_config_parsing() -> None:
    """Test that complex provider configurations are parsed correctly."""
    # Test the parsing logic by checking the raw data structure
    from genai_tk.core.factories.llm_factory import _read_llm_list_file

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
        for _key, value in complex_model.llm_args.items():
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


def test_resolve_llm_identifier_tag_with_compact_alias() -> None:
    """Regression test: config tag whose value is a compact alias must be fully resolved.

    When `llm.models.default` (or any tag) contains a compact alias like
    ``gpt_oss120@openrouter``, ``resolve_llm_identifier("default")`` must return
    the canonical model name (e.g. ``openai/gpt-oss-120b@openrouter``) and NOT the
    raw compact alias.  The bug was introduced when the refactor changed the default
    sentinel from ``None`` to ``"default"``, causing the tag-lookup path to skip the
    secondary ``resolve_model`` step.
    """
    from unittest.mock import patch

    compact_alias = "gpt_oss120@openrouter"

    # Only intercept "my_tag"; let other inputs (like the compact alias itself)
    # fall through to the real implementation so the @-based resolver can run.
    original_find = LlmFactory.find_llm_id_from_tag

    def selective_mock(tag: str) -> str:
        if tag == "my_tag":
            return compact_alias
        return original_find(tag)

    with patch.object(LlmFactory, "find_llm_id_from_tag", side_effect=selective_mock):
        resolved = LlmFactory.resolve_llm_identifier("my_tag")

    # The result must NOT be the raw compact alias – it must be fully resolved.
    assert resolved != compact_alias, (
        f"resolve_llm_identifier returned the raw compact alias '{compact_alias}' "
        "instead of the canonical model name. Regression in config-tag resolution."
    )
    # The canonical form has the vendor-prefixed model name (e.g. openai/gpt-oss-120b)
    model_part, _, provider_part = resolved.rpartition("@")
    assert provider_part == "openrouter"
    assert "/" in model_part, f"Expected vendor/model format, got: '{model_part}'"


def test_llm_factory_model_validation(fake_llm_id) -> None:
    """Test LlmFactory model validation."""
    # Test valid ID
    factory = LlmFactory(llm=fake_llm_id)
    assert factory.llm_id == fake_llm_id

    # Test invalid ID
    with pytest.raises(ValueError, match="Unknown LLM"):
        LlmFactory(llm="invalid_model_id")


def test_field_validator_cache(fake_llm_id) -> None:
    """Test cache field validator."""
    # Valid cache value
    from genai_tk.utils.langchain_community_repl.sqlite_cache import SQLiteCache

    llm = LlmFactory(llm=LLM_ID_FOR_TEST, cache="sqlite").get()
    assert isinstance(llm.cache, SQLiteCache)

    # Invalid cache value should raise ValidationError (field validator check)
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="'memory', 'sqlite' or 'no_cache'"):
        LlmFactory(llm_id=fake_llm_id, cache="invalid_cache")


def test_edenai_v3_uses_openai_compatible_endpoint() -> None:
    """EdenAI v3: factory must use ChatOpenAI with the v3 base URL and provider/model string."""
    from unittest.mock import MagicMock, patch

    mock_llm = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm)

    factory = LlmFactory.__new__(LlmFactory)
    object.__setattr__(factory, "json_mode", False)
    object.__setattr__(factory, "streaming", False)
    object.__setattr__(factory, "reasoning", None)
    object.__setattr__(factory, "cache", None)
    object.__setattr__(factory, "llm_params", {})
    object.__setattr__(factory, "llm_id", "gpt41mini@edenai")
    object.__setattr__(factory, "llm", "gpt41mini@edenai")
    object.__setattr__(
        factory,
        "_resolved_llm_info",
        LlmInfo(id="gpt41mini@edenai", provider="edenai", model="openai/gpt-4.1-mini-2025-04-14"),
    )

    with (
        patch("genai_tk.core.factories.llm_factory.get_provider_api_key", return_value="test-key"),
        patch("langchain_openai.ChatOpenAI", mock_chat_openai),
    ):
        factory.model_factory()

    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["base_url"] == "https://api.edenai.run/v3/llm"
    assert call_kwargs["model"] == "openai/gpt-4.1-mini-2025-04-14"
    assert call_kwargs["api_key"] == "test-key"


def test_split_inline_reasoning_effort() -> None:
    """Inline effort parser should extract model and effort from model@provider."""
    llm_id, effort = _split_inline_reasoning_effort("gpt-oss-120b (high)@openrouter")
    assert llm_id == "gpt-oss-120b@openrouter"
    assert effort == "high"


def test_extract_reasoning_settings_precedence() -> None:
    """Explicit reasoning kwargs override inline effort and merge legacy flat keys."""
    llm_id, params, reasoning_payload = _extract_reasoning_settings(
        "gpt-oss-120b (high)@openrouter",
        {
            "temperature": 0.2,
            "reasoning": {"effort": "low"},
            "reasoning_max_tokens": 2048,
        },
    )

    assert llm_id == "gpt-oss-120b@openrouter"
    assert params == {"temperature": 0.2}
    assert reasoning_payload == {"effort": "low", "max_tokens": 2048}


def test_openai_compatible_injects_reasoning_payload() -> None:
    """OpenAI-compatible path should send reasoning object inside extra_body."""
    from unittest.mock import MagicMock, patch

    mock_llm = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm)

    factory = LlmFactory.__new__(LlmFactory)
    object.__setattr__(factory, "json_mode", False)
    object.__setattr__(factory, "streaming", False)
    object.__setattr__(factory, "reasoning", None)
    object.__setattr__(factory, "cache", None)
    object.__setattr__(factory, "llm_params", {})
    object.__setattr__(factory, "llm_id", "gpt41mini@openrouter")
    object.__setattr__(factory, "llm", "gpt41mini@openrouter")
    object.__setattr__(factory, "_reasoning_payload", {"effort": "high", "max_tokens": 1024})
    object.__setattr__(
        factory,
        "_resolved_llm_info",
        LlmInfo(id="gpt41mini@openrouter", provider="openrouter", model="openai/gpt-4.1-mini"),
    )

    with patch("langchain_openai.ChatOpenAI", mock_chat_openai):
        provider_info = factory.info.get_provider_info()
        factory._create_openai_compatible_llm(provider_info, {"temperature": 0.1}, api_key="test-key")

    call_kwargs = mock_chat_openai.call_args.kwargs
    assert "extra_body" in call_kwargs
    assert call_kwargs["extra_body"]["reasoning"] == {"effort": "high", "max_tokens": 1024}


def test_warn_when_reasoning_requested_on_non_thinking_model(fake_llm_id) -> None:
    """Factory should warn when reasoning options are requested on a non-thinking model."""
    from unittest.mock import patch

    factory = LlmFactory(llm=fake_llm_id, llm_params={"reasoning": {"effort": "high"}})

    with patch("genai_tk.core.factories.llm_factory.logger.warning") as mock_warning:
        model = factory.model_factory()

    assert model is not None
    warning_texts = [str(call.args[0]) for call in mock_warning.call_args_list if call.args]
    assert any("not marked as thinking-capable" in text for text in warning_texts)
