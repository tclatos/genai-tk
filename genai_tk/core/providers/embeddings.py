"""Provider-specific embeddings classes and wrappers."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr


class AzureEmbeddingsWrapper(Embeddings):
    """Wrapper around AzureOpenAIEmbeddings that parses name/version format."""

    @classmethod
    def create(
        cls,
        *,
        model: str,
        api_key: SecretStr | None = None,
        cache: str | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        from langchain_openai import AzureOpenAIEmbeddings

        _ = cache  # Local model cache directory is not used by Azure OpenAI
        name, _, api_version = model.partition("/")
        return AzureOpenAIEmbeddings(
            azure_deployment=name,
            model=name,
            api_version=api_version,
            api_key=api_key or None,
            **kwargs,
        )
