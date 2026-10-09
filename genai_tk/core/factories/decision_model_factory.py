"""Decision models factory and configuration management.

This module provides a factory for instantiating decision models (System One)
across providers (OpenRouter, TypeSafe, fake for testing, or chat model adapters).
"""

from __future__ import annotations

import difflib
from functools import cached_property
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.chat_adapter import ChatModelDecisionModel
from genai_tk.core.decision_models.fake import FakeDecisionModel
from genai_tk.core.decision_models.openrouter import OpenRouterDecisionModel
from genai_tk.core.decision_models.typesafe import TypeSafeDecisionModel
from genai_tk.core.factories.llm_factory import get_llm


class DecisionModelInfo(BaseModel):
    """Metadata describing a configured decision model."""

    id: str
    provider: str
    model: str
    context_length: int | None = None
    max_questions: int | None = None
    description: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"Decision model ID must be in format 'model@provider', got '{v}'")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def model_id(self) -> str:
        return self.id.split("@")[0]


class DecisionModelsConfig(BaseModel):
    """Configuration for decision models in active profile."""

    model_config = ConfigDict(extra="ignore")
    default: str = "clef_flash@openrouter"
    fast: str = "clef_flash@openrouter"
    accurate: str = "jev@openrouter"
    fake: str = "fake_decision@fake"


class DecisionSection(BaseModel):
    """Section in app_conf.yaml under 'decision'."""

    model_config = ConfigDict(extra="ignore")
    models: DecisionModelsConfig = Field(default_factory=DecisionModelsConfig)
    registry: list[dict[str, Any]] = Field(default_factory=list)


def decision_config() -> DecisionSection:
    """Load the decision configuration section."""
    raw = global_config().get("decision", default={})
    if isinstance(raw, (dict, DictConfig)):
        raw_dict = dict(raw)
    else:
        raw_dict = {}

    # Also check if registry was merged under decision_models
    raw_models = global_config().get("decision_models", default={})
    if isinstance(raw_models, (dict, DictConfig)):
        reg = raw_models.get("registry")
        if reg and not raw_dict.get("registry"):
            raw_dict["registry"] = list(reg)

    return DecisionSection.model_validate(raw_dict)


def _read_decision_models_list() -> list[DecisionModelInfo]:
    """Read decision models registry from merged configuration."""
    section = decision_config()
    registry = section.registry
    if not registry:
        return []

    models: list[DecisionModelInfo] = []
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("model_id")
        if not model_id:
            continue
        ctx = entry.get("context_length")
        max_q = entry.get("max_questions")
        desc = entry.get("description")
        providers = entry.get("providers", [])
        for prov_item in providers:
            if isinstance(prov_item, (dict, DictConfig)):
                for provider, model_name in prov_item.items():
                    models.append(
                        DecisionModelInfo(
                            id=f"{model_id}@{provider}",
                            provider=provider,
                            model=str(model_name),
                            context_length=ctx,
                            max_questions=max_q,
                            description=desc,
                        )
                    )
    return models


class DecisionModelFactory(BaseModel):
    """Factory for creating decision models."""

    decision_model: str = "default"
    _cached_items: list[DecisionModelInfo] | None = None

    @classmethod
    def get_known_models(cls) -> list[DecisionModelInfo]:
        return _read_decision_models_list()

    @classmethod
    def get_known_models_dict(cls) -> dict[str, DecisionModelInfo]:
        return {m.id: m for m in cls.get_known_models()}

    @classmethod
    def resolve_model_id(cls, identifier: str) -> str:
        """Resolve tag or alias to model@provider identifier."""
        models_cfg = decision_config().models
        cfg_dict = models_cfg.model_dump()
        if identifier in cfg_dict:
            return cfg_dict[identifier]

        known = cls.get_known_models_dict()
        if identifier in known:
            return identifier

        # Match prefix without provider if unique
        matching = [mid for mid in known if mid.startswith(f"{identifier}@")]
        if len(matching) == 1:
            return matching[0]
        elif len(matching) > 1:
            # Prefer openrouter if multiple providers are configured for same model
            for mid in matching:
                if mid.endswith("@openrouter"):
                    return mid
            return matching[0]

        # Best effort or raise
        closest = difflib.get_close_matches(identifier, list(known.keys()) + list(cfg_dict.keys()), n=1)
        hint = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Unknown decision model identifier '{identifier}'.{hint}")

    @classmethod
    def create(cls, identifier: str = "default", **kwargs: Any) -> BaseDecisionModel:
        """Instantiate a BaseDecisionModel from an identifier or tag."""
        resolved = cls.resolve_model_id(identifier)
        known = cls.get_known_models_dict()
        info = known.get(resolved)

        if not info:
            # Fallback parsing for unknown model@provider
            if "@" in resolved:
                m_part, p_part = resolved.split("@", 1)
                info = DecisionModelInfo(id=resolved, provider=p_part, model=m_part)
            else:
                raise ValueError(f"Cannot instantiate decision model from '{resolved}'")

        if info.max_questions is not None:
            kwargs.setdefault("max_questions", info.max_questions)

        if info.provider == "openrouter":
            return OpenRouterDecisionModel(model=info.model, **kwargs)
        elif info.provider == "typesafe":
            return TypeSafeDecisionModel(model=info.model, **kwargs)
        elif info.provider == "fake":
            return FakeDecisionModel(model=info.id, **kwargs)
        else:
            raise ValueError(f"Unsupported decision model provider: '{info.provider}'")

    @classmethod
    def from_chat_model(cls, llm: BaseChatModel | str, **kwargs: Any) -> BaseDecisionModel:
        """Create a decision model adapter wrapping a standard LangChain BaseChatModel."""
        chat_model = get_llm(llm) if isinstance(llm, str) else llm
        return ChatModelDecisionModel(chat_model=chat_model, **kwargs)


def get_decision_model(
    decision_model: str = "default",
    **kwargs: Any,
) -> BaseDecisionModel:
    """Return a configured BaseDecisionModel.

    Args:
        decision_model: Model ID in 'model@provider' format or tag ('default', 'fast', 'fake', etc.).
        **kwargs: Provider-specific options (e.g. timeout, api_key).
    """
    return DecisionModelFactory.create(decision_model, **kwargs)


def get_decision_model_from_chat_model(
    llm: BaseChatModel | str = "default",
    **kwargs: Any,
) -> BaseDecisionModel:
    """Return a DecisionModel backed by a standard chat LLM via structured prompting."""
    return DecisionModelFactory.from_chat_model(llm, **kwargs)
