"""Provider-specific chat model classes inheriting from LangChain chat models.

Organized with inheritance and mixins to eliminate branching in factory code
and support Inversion of Control (IoC).
"""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

from genai_tk.core.providers.mixins import (
    EdenAIReasoningMixin,
    EdenAIRoutingMixin,
    OpenRouterRoutingMixin,
    ReasoningMixin,
    SeedHandlingMixin,
)


class BaseOpenAIChat(ChatOpenAI):
    """Base OpenAI-compatible chat model for providers exposing OpenAI-compatible endpoints."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        custom_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        seed_location: str | None = "root",
        reasoning_payload: dict[str, Any] | None = None,
        routing_strategy: str | dict[str, Any] | None = None,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        params = (llm_params or {}).copy()
        params.update(kwargs)

        params = SeedHandlingMixin.apply_seed(params, location=seed_location)

        create_params: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            **params,
        }
        if "max_tokens" not in create_params and max_tokens is not None:
            create_params["max_tokens"] = max_tokens
        if base_url:
            create_params["base_url"] = base_url
        if extra_body:
            create_params["extra_body"] = extra_body.copy()
        if custom_headers:
            create_params["default_headers"] = custom_headers.copy()

        cls._configure_extra(create_params, model=model, reasoning=reasoning_payload, routing=routing_strategy)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**create_params)

    @classmethod
    def _configure_extra(
        cls,
        create_params: dict[str, Any],
        model: str,
        reasoning: dict[str, Any] | None,
        routing: str | dict[str, Any] | None,
    ) -> None:
        """Hook for subclasses to configure reasoning, routing, or provider-specific headers."""
        if reasoning:
            ReasoningMixin.apply_reasoning(create_params, reasoning)
        if routing:
            logger.warning(f"Routing '{routing}' not supported by default OpenAI chat wrapper.")


class OpenAIChat(BaseOpenAIChat):
    """Standard OpenAI Chat model."""


class OpenRouterChat(BaseOpenAIChat):
    """OpenRouter chat model supporting routing objectives and reasoning payload."""

    @classmethod
    def _configure_extra(
        cls,
        create_params: dict[str, Any],
        model: str,
        reasoning: dict[str, Any] | None,
        routing: str | dict[str, Any] | None,
    ) -> None:
        OpenRouterRoutingMixin.apply_routing(
            create_params,
            routing_strategy=routing,
            model_name=model,
            extra_body_default=create_params.get("extra_body"),
        )
        if reasoning:
            ReasoningMixin.apply_reasoning(create_params, reasoning)


class EdenAIChat(BaseOpenAIChat):
    """EdenAI chat model supporting EU/global endpoints, routing, and Qwen thinking flags."""

    @classmethod
    def _configure_extra(
        cls,
        create_params: dict[str, Any],
        model: str,
        reasoning: dict[str, Any] | None,
        routing: str | dict[str, Any] | None,
    ) -> None:
        EdenAIRoutingMixin.apply_routing(create_params, routing_strategy=routing)
        if reasoning:
            EdenAIReasoningMixin.apply_reasoning(create_params, reasoning, model_name=model)


class GroqChat(BaseOpenAIChat):
    """Groq chat model with seed placed inside model_kwargs."""

    @classmethod
    def create(cls, **kwargs: Any) -> ChatOpenAI:
        kwargs["seed_location"] = "model_kwargs"
        return super().create(**kwargs)


class VllmChat(BaseOpenAIChat):
    """vLLM and local on-prem OpenAI-compatible server chat model."""


class DeepInfraChat(BaseOpenAIChat):
    """DeepInfra chat model."""


class MistralChat(BaseChatModel):
    """MistralAI chat model via langchain_mistralai."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        api_key: SecretStr | None = None,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_mistralai import ChatMistralAI

        params = (llm_params or {}).copy()
        params.update(kwargs)
        params.pop("seed", None)
        return ChatMistralAI(model_name=model, api_key=api_key, **params)


class AzureChat(BaseChatModel):
    """Azure OpenAI chat model handling name/version parsing and json mode."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        api_key: SecretStr | None = None,
        json_mode: bool = False,
        special_env_vars: dict[str, str] | None = None,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_openai import AzureChatOpenAI

        params = (llm_params or {}).copy()
        params.update(kwargs)
        name, _, api_version = model.partition("/")
        if not api_version:
            special_env = special_env_vars or {}
            api_version = special_env.get("api_version") or os.environ.get("OPENAI_API_VERSION") or ""

        llm = AzureChatOpenAI(
            name=name,
            azure_deployment=name,
            model=name,
            api_version=api_version,
            api_key=api_key,
            **params,
        )
        if json_mode:
            llm = cast(BaseChatModel, llm.bind(response_format={"type": "json_object"}))
        return llm


class OllamaChat(BaseChatModel):
    """Ollama local model with proxy bypass."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        reasoning_payload: dict[str, Any] | None = None,
        base_url: str | None = None,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        original_proxy_env = {}
        proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        for var in proxy_vars:
            if var in os.environ:
                original_proxy_env[var] = os.environ[var]
                del os.environ[var]

        try:
            params = (llm_params or {}).copy()
            params.update(kwargs)
            params.pop("seed", None)
            reasoning_enabled = bool(reasoning_payload) if reasoning_payload is not None else False
            return ChatOllama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                reasoning=reasoning_enabled,
                **params,
            )
        finally:
            for var, value in original_proxy_env.items():
                os.environ[var] = value


class LiteLLMChat(BaseChatModel):
    """LiteLLM router chat model."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_litellm import ChatLiteLLM

        params = (llm_params or {}).copy()
        params.update(kwargs)
        return ChatLiteLLM(model=model, **params)


class HuggingFaceChat(BaseChatModel):
    """HuggingFace endpoint chat model."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            do_sample=False,
            provider="auto",
        )
        return ChatHuggingFace(llm=endpoint)


class FakeChat(BaseChatModel):
    """Deterministic fake chat model for unit tests."""

    @classmethod
    def create(
        cls,
        routing_strategy: str | dict[str, Any] | None = None,
        provider: str = "fake",
        **kwargs: Any,
    ) -> BaseChatModel:
        from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel

        if routing_strategy:
            logger.warning(
                f"Provider routing '{routing_strategy}' is not supported by provider '{provider}'. Ignoring routing."
            )

        return ParrotFakeChatModel()
