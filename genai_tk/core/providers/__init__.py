"""Provider configuration and API key management.

This module contains shared provider configurations and utilities for
managing API keys across different AI service providers.

It also defines the concept of a **lab** — the AI research organisation that
*created* a model, which can be independent of the *provider* (the API
endpoint used to access it).  For example, Anthropic models can be accessed
via the direct ``anthropic`` provider or via the ``openrouter`` gateway; in
both cases the lab is ``anthropic``.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from genai_tk.config_mgmt.config_mngr import QualifiedClassName
from genai_tk.config_mgmt.import_utils import ImportResolver
from genai_tk.utils.singleton import once

DEEPSEEK_API_BASE = "https://api.deepseek.com"


class ProviderInfo(BaseModel):
    """Structured information about an LLM provider.

    Attributes:
        chat_class: Qualified class for chat models (None if provider doesn't support chat)
        embeddings_class: Optional qualified class for embeddings models
        decision_class: Optional qualified class for decision models
        api_key_env_var: Environment variable name for API key
        api_base: Optional API base URL for OpenAI-compatible providers
        litellm_prefix: LiteLLM provider prefix; null means no prefix (openai-style)
        gateway: True for providers that accept vendor-prefixed model names
        extra_body: Optional extra fields to pass to API (e.g. for OpenRouter quantization)
        special_env_vars: Additional environment variables needed (e.g. AZURE_OPENAI_API_VERSION)
        openai_compatible: True if provider uses OpenAI-compatible API (defaults to detecting from chat_class)
        seed_param_location: Where to place seed parameter ('root' for root params, 'model_kwargs' for groq, None to omit)
        custom_headers: Custom headers to send with API requests
    """

    chat_class: QualifiedClassName | None = Field(None, description="Qualified class for chat models")
    embeddings_class: QualifiedClassName | None = Field(None, description="Qualified class for embeddings models")
    decision_class: QualifiedClassName | None = Field(None, description="Qualified class for decision models")
    api_key_env_var: str = ""
    api_base: str | None = None
    litellm_prefix: str | None = None
    gateway: bool = False
    extra_body: dict[str, Any] | None = None
    special_env_vars: dict[str, str] | None = None
    openai_compatible: bool | None = None
    seed_param_location: str | None = "root"
    custom_headers: dict[str, str] | None = None

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def is_openai_compatible(self) -> bool:
        """Check if provider uses OpenAI-compatible API."""
        if self.openai_compatible is not None:
            return self.openai_compatible
        target = self.chat_class or ""
        return (
            "ChatOpenAI" in target or "BaseOpenAIChat" in target or "OpenAIChat" in target or self.api_base is not None
        )

    def get_special_env_vars(self) -> dict[str, str]:
        """Get special environment variables needed for this provider."""
        result = {}
        if self.special_env_vars:
            for key, env_var in self.special_env_vars.items():
                if env_var in os.environ:
                    result[key] = os.environ[env_var]
        return result

    @property
    def module(self) -> str:
        """Extract module path from chat_class, embeddings_class, or decision_class."""
        target = self.chat_class or self.embeddings_class or self.decision_class or ""
        return ImportResolver.get_module(target) if target else ""

    @property
    def langchain_class(self) -> str:
        """Extract class name from chat_class, embeddings_class, or decision_class."""
        target = self.chat_class or self.embeddings_class or self.decision_class or ""
        return ImportResolver.get_object_name(target) if target else ""

    def get_use_string(self) -> str:
        """Get the chat class string in module:ClassName format."""
        if self.chat_class:
            return self.chat_class
        raise ValueError("Provider has no chat_class configured.")


# ── Lab (model-creator) information ────────────────────────────────────────────


class LabInfo(BaseModel):
    """Information about an AI research lab that creates models.

    A lab is the organisation that *created* a model, independent of the
    *provider* API used to access it (e.g. Anthropic models are available via
    the ``anthropic`` direct provider and also via ``openrouter``; both map to
    the ``anthropic`` lab).

    Attributes:
        id: Unique lab key (e.g. ``openai``, ``anthropic``).
        display_name: Human-readable lab name (e.g. ``"OpenAI"``, ``"Google DeepMind"``).
        vendor_prefixes: Vendor prefixes used in gateway model IDs
            (e.g. ``["meta-llama"]`` matches ``meta-llama/llama-3.1-8b``).
        model_patterns: Case-insensitive substrings matched against the bare
            model name (e.g. ``["claude"]`` matches ``claude-3-5-sonnet``).
        direct_provider: Provider key whose *entire* model catalogue belongs to
            this lab (e.g. ``"anthropic"``).  Avoids pattern matching for
            single-lab direct providers.
    """

    id: str
    display_name: str
    vendor_prefixes: list[str] = []
    model_patterns: list[str] = []
    direct_provider: str | None = None

    model_config = {"frozen": True}

    def matches(self, model_id: str, provider_id: str) -> bool:
        """Return True if this lab is the likely creator of *model_id* at *provider_id*.

        Checks (in order):
        1. direct_provider match — the whole provider belongs to this lab.
        2. vendor prefix — the model ID starts with a known gateway prefix
           (``vendor/model`` format used by OpenRouter, DeepInfra, etc.).
        3. substring patterns — any pattern appears in the model name.
        """
        if self.direct_provider and provider_id == self.direct_provider:
            return True
        # vendor prefix (gateway-style "vendor/model")
        if "/" in model_id:
            prefix = model_id.split("/", 1)[0].lower()
            if any(prefix == vp.lower() for vp in self.vendor_prefixes):
                return True
        # substring pattern on the bare model name
        bare = model_id.split("/")[-1].lower()
        return any(pat.lower() in bare for pat in self.model_patterns)


@once
def _load_provider_info_from_yaml() -> dict[str, ProviderInfo]:
    """Load provider information from YAML config file."""
    from importlib.resources import files as _pkg_files

    # Load from the bundled package data (works both in editable installs and wheels)
    try:
        src = _pkg_files("genai_tk") / "default_config" / "providers" / "providers.yaml"
        yaml_text = src.read_text(encoding="utf-8")
    except Exception as e:
        # Fallback for editable installs where the symlink may not resolve via importlib
        config_path = Path(__file__).parent.parent / "default_config" / "providers" / "providers.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Provider config file not found: {config_path}") from e
        yaml_text = config_path.read_text(encoding="utf-8")

    data = yaml.safe_load(yaml_text)

    providers = {}
    for name, info in data["providers"].items():
        providers[name] = ProviderInfo(
            chat_class=info.get("chat_class") or info.get("use"),
            embeddings_class=info.get("embeddings_class"),
            decision_class=info.get("decision_class"),
            api_key_env_var=info.get("api_key_env_var", ""),
            api_base=info.get("api_base"),
            litellm_prefix=info.get("litellm_prefix"),
            gateway=info.get("gateway", False),
            extra_body=info.get("extra_body"),
            special_env_vars=info.get("special_env_vars"),
            openai_compatible=info.get("openai_compatible"),
            seed_param_location=info.get("seed_param_location", "root"),
            custom_headers=info.get("custom_headers"),
        )

    return providers


@once
def _load_labs_from_yaml() -> dict[str, LabInfo]:
    """Load lab definitions from the same providers YAML config file."""
    from importlib.resources import files as _pkg_files

    try:
        src = _pkg_files("genai_tk") / "default_config" / "providers" / "providers.yaml"
        yaml_text = src.read_text(encoding="utf-8")
    except Exception:
        config_path = Path(__file__).parent.parent / "default_config" / "providers" / "providers.yaml"
        yaml_text = config_path.read_text(encoding="utf-8")

    data = yaml.safe_load(yaml_text)
    labs: dict[str, LabInfo] = {}
    for lab_id, info in (data.get("labs") or {}).items():
        labs[lab_id] = LabInfo(
            id=lab_id,
            display_name=info.get("display_name", lab_id),
            vendor_prefixes=info.get("vendor_prefixes", []),
            model_patterns=info.get("model_patterns", []),
            direct_provider=info.get("direct_provider"),
        )
    return labs


# List of implemented LLM providers, loaded from YAML configuration
PROVIDER_INFO: dict[str, ProviderInfo] = _load_provider_info_from_yaml()

# Known AI labs (model creators), loaded from YAML configuration
LAB_INFO: dict[str, LabInfo] = _load_labs_from_yaml()


def get_lab_for_model(model_id: str, provider_id: str) -> str | None:
    """Return the lab key for *model_id* served via *provider_id*, or ``None``.

    Tries each lab in definition order and returns the first match.

    Args:
        model_id: The bare model name as used by the provider (e.g.
            ``"claude-3-5-sonnet-20241022"`` or
            ``"anthropic/claude-3-5-sonnet"``).
        provider_id: The provider key (e.g. ``"anthropic"``, ``"openrouter"``).

    Returns:
        Lab key string (e.g. ``"anthropic"``) or ``None`` if unknown.
    """
    for lab in LAB_INFO.values():
        if lab.matches(model_id, provider_id):
            return lab.id
    return None


def get_provider_api_env_var(provider: str) -> str | None:
    """Get the environment variable name for a given AI provider's API key.

    Args:
        provider: Name of the AI provider (e.g. "openai", "google")

    Returns:
        The environment variable name if configured (can be empty string), None otherwise

    """
    if provider not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider}. Valid providers are: {list(PROVIDER_INFO.keys())}")
    return PROVIDER_INFO[provider].api_key_env_var


def get_provider_api_key(provider: str) -> SecretStr | None:
    """Get the API key for a given AI provider.

    Args:
        provider: Name of the AI provider (e.g. "openai", "google")

    Returns:
        The API key as SecretStr if found, None otherwise
    """

    # Strip any surrounding quotes and whitespace
    env_var = get_provider_api_env_var(provider)
    if env_var:
        r = os.environ[env_var].strip("\"' \t\n\r")
        return SecretStr(r)
    else:
        return None


def get_provider_info(provider: str) -> ProviderInfo:
    """Get the full ProviderInfo for a given provider.

    Checks global configuration first (allows dynamic application-level extension),
    then falls back to default PROVIDER_INFO.

    Args:
        provider: Name of the AI provider (e.g. "openai", "google", "vllm")

    Returns:
        ProviderInfo object with all provider configuration

    Raises:
        ValueError: If provider is unknown
    """
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        app_providers = global_config().get("providers", default={})
        if isinstance(app_providers, (dict, DictConfig)) and provider in app_providers:
            info = app_providers[provider]
            if isinstance(info, (dict, DictConfig)):
                return ProviderInfo(
                    chat_class=info.get("chat_class") or info.get("use"),
                    embeddings_class=info.get("embeddings_class"),
                    decision_class=info.get("decision_class"),
                    api_key_env_var=info.get("api_key_env_var", ""),
                    api_base=info.get("api_base"),
                    litellm_prefix=info.get("litellm_prefix"),
                    gateway=info.get("gateway", False),
                    extra_body=info.get("extra_body"),
                    special_env_vars=info.get("special_env_vars"),
                    openai_compatible=info.get("openai_compatible"),
                    seed_param_location=info.get("seed_param_location", "root"),
                    custom_headers=info.get("custom_headers"),
                )
    except Exception:
        pass

    if provider not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider}. Valid providers are: {list(PROVIDER_INFO.keys())}")
    return PROVIDER_INFO[provider]
