"""Tests for locally cached model catalogues."""

import json
from unittest.mock import Mock, patch

from genai_tk.core.factories.llm_factory import resolve_model
from genai_tk.core.models_db import (
    EDENAI_CACHE_FILENAME,
    EDENAI_EU_CACHE_FILENAME,
    EDENAI_EU_MODELS_URL,
    EDENAI_MODELS_URL,
    ModelRegion,
    ModelsDb,
)

EDENAI_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "scaleway/glm-5.2",
            "object": "model",
            "owned_by": "scaleway",
            "model_name": "glm-5.2",
            "context_length": 131072,
            "capabilities": {
                "input_modalities": ["text", "image", "pdf", "audio", "video"],
                "output_modalities": ["text"],
                "supports_reasoning": True,
                "supports_function_calling": True,
                "supports_response_schema": True,
            },
            "pricing": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002},
            "regions": [{"code": "eu", "name": "Europe"}],
        },
        {
            "id": "nebius/zai-org/GLM-5.2",
            "object": "model",
            "owned_by": "nebius",
            "capabilities": {"reasoning": True},
            "regions": [{"code": "us", "name": "United States"}],
        },
        {
            "id": "cloudflare/@cf/zai-org/glm-5.2",
            "object": "model",
            "owned_by": "cloudflare",
            "capabilities": {"tool_calling": True},
            "regions": [{"code": "global", "name": "Global"}],
        },
        {"id": "azure/gpt-5.2-codex", "object": "model", "owned_by": "openai", "capabilities": {}},
        {"id": "scaleway/glm-5.1", "object": "model", "owned_by": "scaleway", "capabilities": {}},
    ],
}

EDENAI_EU_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "scaleway/glm-5.2",
            "object": "model",
            "owned_by": "scaleway",
            "capabilities": {"supports_reasoning": True},
            "regions": [{"code": "eu", "name": "Europe"}],
        },
        {"id": "qwen/glm-5.2", "object": "model", "owned_by": "qwen", "capabilities": {}},
    ],
}


def _models_dev_cache(tmp_path):
    cache_path = tmp_path / "models_dev.json"
    cache_path.write_text(json.dumps({}), encoding="utf-8")
    return cache_path


def test_load_merges_cached_edenai_models_and_regions(tmp_path) -> None:
    """Cached EdenAI entries use the same lookup API as models.dev entries."""
    cache_path = _models_dev_cache(tmp_path)
    (tmp_path / EDENAI_CACHE_FILENAME).write_text(json.dumps(EDENAI_MODELS), encoding="utf-8")

    entry = ModelsDb().load(cache_path).lookup("edenai", "scaleway/glm-5.2")

    assert entry is not None
    assert entry.regions == [ModelRegion(code="eu", name="Europe")]
    assert entry.capabilities == ["vision", "thinking", "structured_outputs", "pdf", "audio", "video"]
    assert entry.tool_call is True
    assert entry.context == 131072
    assert entry.cost_input == 1.0
    assert entry.cost_output == 2.0


def test_load_merges_cached_edenai_eur_models_under_its_provider(tmp_path, monkeypatch) -> None:
    """EU cache entries are isolated from the global EdenAI provider catalogue."""
    cache_path = _models_dev_cache(tmp_path)
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)
    (tmp_path / EDENAI_EU_CACHE_FILENAME).write_text(json.dumps(EDENAI_EU_MODELS), encoding="utf-8")

    db = ModelsDb().load(cache_path)

    assert db.lookup("edenai-eur", "scaleway/glm-5.2") is not None
    assert db.lookup("edenai", "scaleway/glm-5.2") is None


def test_load_does_not_fetch_edenai_without_api_key(tmp_path, monkeypatch) -> None:
    """An absent EdenAI cache must not trigger an unauthenticated HTTP request."""
    cache_path = _models_dev_cache(tmp_path)
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)

    with patch("genai_tk.core.models_db.httpx.get") as get:
        ModelsDb().load(cache_path)

    get.assert_not_called()


def test_fetch_edenai_caches_models_when_api_key_is_set(tmp_path, monkeypatch) -> None:
    """Authenticated EdenAI responses are persisted alongside models.dev data."""
    monkeypatch.setenv("EDENAI_API_KEY", "test-key")
    response = Mock()
    response.json.return_value = EDENAI_MODELS

    with patch("genai_tk.core.models_db.httpx.get", return_value=response) as get:
        db = ModelsDb()
        assert db.fetch_edenai(tmp_path / EDENAI_CACHE_FILENAME)

    get.assert_called_once_with(
        EDENAI_MODELS_URL,
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert db.lookup("edenai", "cloudflare/@cf/zai-org/glm-5.2") is not None
    assert (tmp_path / EDENAI_CACHE_FILENAME).exists()


def test_fetch_edenai_eur_caches_eu_models_when_api_key_is_set(tmp_path, monkeypatch) -> None:
    """The European endpoint persists its restricted catalogue independently."""
    monkeypatch.setenv("EDENAI_API_KEY", "test-key")
    response = Mock()
    response.json.return_value = EDENAI_EU_MODELS

    with patch("genai_tk.core.models_db.httpx.get", return_value=response) as get:
        db = ModelsDb()
        assert db.fetch_edenai(tmp_path / EDENAI_EU_CACHE_FILENAME, provider_id="edenai-eur")

    get.assert_called_once_with(
        EDENAI_EU_MODELS_URL,
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert db.lookup("edenai-eur", "scaleway/glm-5.2") is not None
    assert (tmp_path / EDENAI_EU_CACHE_FILENAME).exists()


def test_resolve_edenai_model_lists_all_inference_backends_first() -> None:
    """EdenAI alternatives group a model's inference backends before prior versions."""
    db = ModelsDb()
    db._build_index({})
    db._merge_edenai_models("edenai", EDENAI_MODELS)

    with patch("genai_tk.core.factories.llm_factory.get_models_db", return_value=db):
        model_id, _entry, alternatives = resolve_model("glm-5.2", "edenai")
        cloudflare_model_id, _entry, cloudflare_alternatives = resolve_model("glm-5.2-cloudflare", "edenai")

    glm_52_backends = [candidate for candidate, _score in alternatives[:3]]
    assert model_id in glm_52_backends
    assert glm_52_backends == [
        "scaleway/glm-5.2",
        "nebius/zai-org/GLM-5.2",
        "cloudflare/@cf/zai-org/glm-5.2",
    ]
    assert alternatives[3][0] == "scaleway/glm-5.1"
    assert cloudflare_model_id == "cloudflare/@cf/zai-org/glm-5.2"
    assert cloudflare_alternatives[0][0] == cloudflare_model_id
    assert {candidate for candidate, _score in cloudflare_alternatives[:3]} == set(glm_52_backends)
    assert cloudflare_alternatives[3][0] == "scaleway/glm-5.1"


def test_resolve_edenai_eur_model_uses_eu_catalogue() -> None:
    """European aliases resolve from the EU-filtered catalogue, not OpenRouter."""
    db = ModelsDb()
    db._build_index({})
    db._merge_edenai_models("edenai-eur", EDENAI_EU_MODELS)

    with patch("genai_tk.core.factories.llm_factory.get_models_db", return_value=db):
        model_id, entry, alternatives = resolve_model("glm-5.2-scaleway", "edenai-eur")

    assert model_id == "scaleway/glm-5.2"
    assert entry is not None
    assert entry.provider_id == "edenai-eur"
    assert alternatives[0][0] == "scaleway/glm-5.2"
