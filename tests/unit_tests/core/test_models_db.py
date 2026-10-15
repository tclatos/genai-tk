"""Tests for locally cached model catalogues."""

import json
from unittest.mock import patch

from genai_tk.core.factories.llm_factory import resolve_model
from genai_tk.core.models_db import ModelEntry, ModelRegion, ModelsDb

SAMPLE_MODELS_DEV = {
    "edenai": {
        "id": "edenai",
        "name": "Eden AI",
        "models": {
            "scaleway/glm-5.2": {
                "id": "scaleway/glm-5.2",
                "name": "GLM-5.2 (Scaleway)",
                "family": "glm",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "modalities": {
                    "input": ["text", "image", "pdf", "audio", "video"],
                    "output": ["text"],
                },
                "limit": {"context": 131072, "output": 4096},
                "cost": {"input": 1.0, "output": 2.0},
                "regions": [{"code": "eu", "name": "Europe"}],
            },
            "nebius/zai-org/GLM-5.2": {
                "id": "nebius/zai-org/GLM-5.2",
                "name": "GLM-5.2 (Nebius)",
                "reasoning": True,
                "regions": [{"code": "us", "name": "United States"}],
            },
            "cloudflare/@cf/zai-org/glm-5.2": {
                "id": "cloudflare/@cf/zai-org/glm-5.2",
                "name": "GLM-5.2 (Cloudflare)",
                "tool_call": True,
                "regions": [{"code": "global", "name": "Global"}],
            },
            "scaleway/glm-5.1": {
                "id": "scaleway/glm-5.1",
                "name": "GLM-5.1 (Scaleway)",
            },
        },
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
            "gpt-4o-mini": {
                "id": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "limit": {"context": 128000, "output": 16384},
                "cost": {"input": 0.15, "output": 0.6},
            }
        },
    },
}


def _models_dev_cache(tmp_path):
    cache_path = tmp_path / "models_dev.json"
    cache_path.write_text(json.dumps(SAMPLE_MODELS_DEV), encoding="utf-8")
    return cache_path


def test_load_and_lookup_models_dev_entry(tmp_path) -> None:
    """Models.dev entries (including EdenAI) are loaded into the normalized index."""
    cache_path = _models_dev_cache(tmp_path)
    db = ModelsDb().load(cache_path)

    entry = db.lookup("edenai", "scaleway/glm-5.2")
    assert entry is not None
    assert entry.regions == [ModelRegion(code="eu", name="Europe")]
    assert entry.capabilities == ["vision", "thinking", "structured_outputs", "pdf", "audio", "video"]
    assert entry.tool_call is True
    assert entry.context == 131072
    assert entry.output == 4096
    assert entry.cost_input == 1.0
    assert entry.cost_output == 2.0


def test_lookup_edenai_eur_aliases_to_edenai(tmp_path) -> None:
    """The European EdenAI provider aliases to the models.dev edenai catalogue."""
    cache_path = _models_dev_cache(tmp_path)
    db = ModelsDb().load(cache_path)

    assert db.lookup("edenai-eur", "scaleway/glm-5.2") is not None
    assert db.lookup("edenai", "scaleway/glm-5.2") is not None


def test_provider_models_edenai_eur_aliases_to_edenai(tmp_path) -> None:
    """provider_models for edenai-eur returns the models configured under edenai."""
    cache_path = _models_dev_cache(tmp_path)
    db = ModelsDb().load(cache_path)

    eur_models = db.provider_models("edenai-eur")
    global_models = db.provider_models("edenai")
    assert eur_models == global_models
    assert "scaleway/glm-5.2" in eur_models


def test_lookup_vendor_fallback(tmp_path) -> None:
    """Vendor-prefixed IDs fall back to the vendor's direct section."""
    cache_path = _models_dev_cache(tmp_path)
    db = ModelsDb().load(cache_path)

    entry = db.lookup("openrouter", "openai/gpt-4o-mini")
    assert entry is not None
    assert entry.id == "gpt-4o-mini"
    assert entry.provider_id == "openai"


def test_model_entry_validators() -> None:
    """Field validators normalize 0 to None and string regions to ModelRegion."""
    entry = ModelEntry(
        id="test-model",
        name="Test",
        provider_id="test",
        context=0,
        output=0,
        regions=["eu", {"code": "us", "name": "United States"}],
    )
    assert entry.context is None
    assert entry.output is None
    assert entry.regions == [
        ModelRegion(code="eu", name="eu"),
        ModelRegion(code="us", name="United States"),
    ]


def test_resolve_edenai_model_lists_all_inference_backends_first() -> None:
    """EdenAI alternatives group a model's inference backends before prior versions."""
    db = ModelsDb()
    db._build_index(SAMPLE_MODELS_DEV)

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


def test_resolve_edenai_eur_model_uses_edenai_catalogue() -> None:
    """European aliases resolve from the unified EdenAI catalogue in models.dev."""
    db = ModelsDb()
    db._build_index(SAMPLE_MODELS_DEV)

    with patch("genai_tk.core.factories.llm_factory.get_models_db", return_value=db):
        model_id, entry, alternatives = resolve_model("glm-5.2-cloudflare", "edenai-eur")

    assert model_id == "cloudflare/@cf/zai-org/glm-5.2"
    assert entry is not None
    assert entry.provider_id == "edenai"
    assert alternatives[0][0] == "cloudflare/@cf/zai-org/glm-5.2"
