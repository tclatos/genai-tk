"""Provider configuration and API key management.

This module contains shared provider configurations and utilities for
managing API keys across different AI service providers.

"""

import os
from dataclasses import dataclass

from pydantic import SecretStr

OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
DEEPSEEK_API_BASE = "https://api.deepseek.com"


@dataclass(frozen=True)
class ProviderInfo:
    """Structured information about an LLM provider.

    Attributes:
        module: Python module name to import
        langchain_class: LangChain class name (e.g., "ChatOpenAI")
        api_key_env_var: Environment variable name for API key (empty string if not required)
        api_base: Optional API base URL for OpenAI-compatible providers
    """

    module: str
    langchain_class: str
    api_key_env_var: str
    api_base: str | None = None

    def get_use_string(self) -> str:
        """Get the 'use' string module:ClassName  format."""
        return f"{self.module}:{self.langchain_class}"


# List of implemented LLM providers, with structured configuration
PROVIDER_INFO: dict[str, ProviderInfo] = {
    "fake": ProviderInfo("langchain_core.language_models.fake_chat_models", "ParrotFakeChatModel", ""),
    "openai": ProviderInfo("langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"),
    "deepinfra": ProviderInfo(
        "langchain_openai", "ChatOpenAI", "DEEPINFRA_API_TOKEN", "https://api.deepinfra.com/v1/openai"
    ),
    "groq": ProviderInfo("langchain_openai", "ChatOpenAI", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "ollama": ProviderInfo("langchain_ollama", "ChatOllama", ""),
    "edenai": ProviderInfo("langchain_openai", "ChatOpenAI", "EDENAI_API_KEY", "https://api.edenai.run/v2/llm"),
    "azure": ProviderInfo("langchain_openai", "AzureChatOpenAI", "AZURE_OPENAI_API_KEY"),
    "together": ProviderInfo("langchain_together", "ChatTogether", "TOGETHER_API_KEY"),
    "deepseek": ProviderInfo("langchain_deepseek", "ChatDeepSeek", "DEEPSEEK_API_KEY"),
    "openrouter": ProviderInfo("langchain_openai", "ChatOpenAI", "OPENROUTER_API_KEY", OPENROUTER_API_BASE),
    "huggingface": ProviderInfo("langchain_openai", "ChatOpenAI", "HUGGINGFACEHUB_API_TOKEN"),
    "mistralai": ProviderInfo("langchain_openai", "ChatOpenAI", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    "litellm": ProviderInfo("litellm", "ChatLiteLLM", ""),
    # NOT TESTED:
    "bedrock": ProviderInfo("langchain_aws", "ChatBedrock", "AWS_ACCESS_KEY_ID"),
    "anthropic": ProviderInfo("langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
    "google": ProviderInfo("langchain_google_genai", "ChatGoogleGenerativeAI", "GOOGLE_API_KEY"),
    "custom": ProviderInfo("langchain_openai", "ChatOpenAI", ""),
}


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
