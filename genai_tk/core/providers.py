"""Provider configuration and API key management.

This module contains shared provider configurations and utilities for
managing API keys across different AI service providers.

"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr

from genai_tk.utils.config_mngr import QualifiedClassName, get_module_from_qualified, get_object_name_from_qualified
from genai_tk.utils.singleton import once

DEEPSEEK_API_BASE = "https://api.deepseek.com"


class ProviderInfo(BaseModel):
    """Structured information about an LLM provider.

    Attributes:
        use: Combined module.ClassName string (e.g., 'langchain_openai.ChatOpenAI')
        api_key_env_var: Environment variable name for API key
        api_base: Optional API base URL for OpenAI-compatible providers
        litellm_prefix: LiteLLM provider prefix; null means no prefix (openai-style)
        gateway: True for providers that accept vendor-prefixed model names
        extra_body: Optional extra fields to pass to API (e.g. for OpenRouter quantization)
        special_env_vars: Additional environment variables needed (e.g. AZURE_OPENAI_API_VERSION)
        openai_compatible: True if provider uses OpenAI-compatible API (defaults to detecting from 'use')
        seed_param_location: Where to place seed parameter ('root' for root params, 'model_kwargs' for groq, None to omit)
        custom_headers: Custom headers to send with API requests
    """

    use: QualifiedClassName = Field(..., description="Module and class in format 'module.path.ClassName'")
    api_key_env_var: str
    api_base: str | None = None
    litellm_prefix: str | None = None
    gateway: bool = False
    extra_body: dict[str, Any] | None = None
    special_env_vars: dict[str, str] | None = None
    openai_compatible: bool | None = None
    seed_param_location: str | None = "root"
    custom_headers: dict[str, str] | None = None

    model_config = {"frozen": True}

    def is_openai_compatible(self) -> bool:
        """Check if provider uses OpenAI-compatible API."""
        if self.openai_compatible is not None:
            return self.openai_compatible
        # Auto-detect: if using ChatOpenAI or has api_base, it's OpenAI-compatible
        return "ChatOpenAI" in self.langchain_class or self.api_base is not None

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
        """Extract module path from use string."""
        return get_module_from_qualified(self.use)

    @property
    def langchain_class(self) -> str:
        """Extract class name from use string."""
        return get_object_name_from_qualified(self.use)

    def get_use_string(self) -> str:
        """Get the 'use' string in module:ClassName format."""
        return self.use


@once
def _load_provider_info_from_yaml() -> dict[str, ProviderInfo]:
    """Load provider information from YAML config file."""
    # Look for providers.yaml in config/basic/providers/ directory
    config_path = Path(__file__).parent.parent.parent / "config" / "basic" / "providers" / "providers.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Provider config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    providers = {}
    for name, info in data["providers"].items():
        providers[name] = ProviderInfo(
            use=info["use"],
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


# List of implemented LLM providers, loaded from YAML configuration
PROVIDER_INFO: dict[str, ProviderInfo] = _load_provider_info_from_yaml()


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

    Args:
        provider: Name of the AI provider (e.g. "openai", "google")

    Returns:
        ProviderInfo object with all provider configuration

    Raises:
        ValueError: If provider is unknown
    """
    if provider not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider}. Valid providers are: {list(PROVIDER_INFO.keys())}")
    return PROVIDER_INFO[provider]
