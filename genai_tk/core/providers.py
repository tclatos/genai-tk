"""Provider configuration and API key management.

This module contains shared provider configurations and utilities for
managing API keys across different AI service providers.

"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator

OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
DEEPSEEK_API_BASE = "https://api.deepseek.com"


class ProviderInfo(BaseModel):
    """Structured information about an LLM provider.

    Attributes:
        use: Combined module:ClassName string (e.g., 'langchain_openai:ChatOpenAI')
        api_key_env_var: Environment variable name for API key
        api_base: Optional API base URL for OpenAI-compatible providers
    """

    use: str = Field(..., description="Module and class in format 'module.path:ClassName'")
    api_key_env_var: str
    api_base: str | None = None

    model_config = {"frozen": True}

    @field_validator("use")
    @classmethod
    def validate_use_format(cls, v: str) -> str:
        """Validate that use field contains module:ClassName format."""
        if ":" not in v:
            raise ValueError(f"'use' field must be in format 'module:ClassName', got: {v}")
        return v

    @property
    def module(self) -> str:
        """Extract module path from use string."""
        return self.use.split(":")[0]

    @property
    def langchain_class(self) -> str:
        """Extract class name from use string."""
        return self.use.split(":")[1]

    def get_use_string(self) -> str:
        """Get the 'use' string in module:ClassName format."""
        return self.use


@lru_cache(maxsize=1)
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
