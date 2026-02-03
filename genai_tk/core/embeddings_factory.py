"""Embedding models factory and management.

This module provides a comprehensive factory pattern implementation for creating
and managing embedding models from various providers. It supports a wide range
of embedding technologies across cloud-based and local CPU-based models.

Key Features:
- Unified interface for multiple embedding providers
- Dynamic configuration through configuration files
- Secure API key management
- Flexible model caching and persistence
- Seamless integration with vector stores and machine learning workflows

Supported Providers:
- OpenAI
- Google Generative AI
- HuggingFace
- EdenAI
- Azure OpenAI
- Ollama
- OpenRouter
- DeepInfra

Example:
    # Get default embeddings
    embeddings = get_embeddings()

    # Get specific model
    embeddings = get_embeddings(embeddings="huggingface_all-mpnet-base-v2")
    vectors = embeddings.embed_documents(["Sample text"])
"""

import os  # noqa: I001
from functools import cached_property, lru_cache
from typing import Annotated
import yaml
from devtools import debug  # noqa: F401
from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger
from pydantic import BaseModel, Field, computed_field

from genai_tk.extra.kv_store_registry import KvStoreRegistry
from genai_tk.utils.config_mngr import global_config
from genai_tk.core.providers import get_provider_api_env_var, get_provider_api_key

_ = load_dotenv(verbose=True)


class EmbeddingsInfo(BaseModel):
    """Information about an embeddings model.

    Provides comprehensive details about an embedding model, including
    its unique identifier, constructor, provider, and optional API key.

    Attributes:
        id: Unique identifier for the embeddings model
        provider: Name of the provider
        model: Provider-specific model name
        api_key_env_var: Optional API key for accessing the model
        prefix: Optional prefix required by some models in API calls
        dimension: Dimension of the embeddings model
    """

    id: str
    provider: str
    model: str
    prefix: str = ""
    dimension: int | None = None

    def __hash__(self) -> int:
        return hash(self.id)


def _read_embeddings_list_file() -> list[EmbeddingsInfo]:
    """Read embeddings configuration from YAML file.

    Returns:
        List of configured embeddings models

    Raises:
        AssertionError: If configuration file is not found
    """
    yml_file = global_config().get_file_path("embeddings.list")
    with open(yml_file) as f:
        data = yaml.safe_load(f)
    embeddings = []
    if "embeddings" in data:
        for model_entry in data["embeddings"]:
            model_id = model_entry["model"]["id"]
            dimension = model_entry.get("dimension")
            for provider_info in model_entry["providers"]:
                for provider, model_name in provider_info.items():
                    embedding_info = {
                        "id": f"{model_id}_{provider}",
                        "provider": provider,
                        "model": model_name,
                        "dimension": dimension,
                    }
                    embeddings.append(EmbeddingsInfo(**embedding_info))
    return embeddings


class EmbeddingsFactory(BaseModel):
    """Factory for creating and managing embeddings models.

    Provides a flexible and configurable way to instantiate embedding models
    from various providers with support for caching and dynamic configuration.

    Attributes:
        embeddings: Unified embeddings identifier (can be either ID or tag) - recommended
        embeddings_id: (Deprecated) Unique identifier for the embeddings model
        embeddings_tag: (Deprecated) Embeddings tag from config (e.g., 'local', 'default')
        encoding_str: Optional encoding configuration string
        retrieving_str: Optional retrieving configuration string
        cache_embeddings: cache embeddings in a KV store
    """

    embeddings: Annotated[str | None, Field(validate_default=True)] = None
    embeddings_id: Annotated[str | None, Field(validate_default=True)] = None
    embeddings_tag: str | None = None
    encoding_str: str | None = None
    retrieving_str: str | None = None
    cache_embeddings: bool = False

    @computed_field
    @cached_property
    def info(self) -> EmbeddingsInfo:
        """Retrieve embeddings model information.

        Returns:
            Configuration details for the selected embeddings model
        """
        assert self.embeddings_id
        return EmbeddingsFactory.known_items_dict().get(self.embeddings_id)  # type: ignore

    def model_post_init(self, __context: dict) -> None:
        """Post-initialization validation and tag resolution."""
        # Handle deprecation warnings
        if self.embeddings_id is not None:
            logger.warning(
                "⚠️  'embeddings_id' parameter is deprecated. Use 'embeddings' instead. "
                "Example: EmbeddingsFactory(embeddings='ada_002_openai') or EmbeddingsFactory(embeddings='local')"
            )

        if self.embeddings_tag is not None:
            logger.warning(
                "⚠️  'embeddings_tag' parameter is deprecated. Use 'embeddings' instead. "
                "Example: EmbeddingsFactory(embeddings='local')"
            )

        # Check for conflicting parameters
        unified_params_count = sum(
            [self.embeddings is not None, self.embeddings_id is not None, self.embeddings_tag is not None]
        )
        if unified_params_count > 1:
            raise ValueError(
                "Cannot specify multiple embeddings selection parameters. Use 'embeddings' parameter only (recommended)"
            )

        # Resolve unified 'embeddings' parameter
        if self.embeddings is not None:
            resolved_id = EmbeddingsFactory.resolve_embeddings_identifier(self.embeddings)
            object.__setattr__(self, "embeddings_id", resolved_id)

        # Handle legacy embeddings_tag parameter
        elif self.embeddings_tag is not None:
            resolved_id = EmbeddingsFactory.find_embeddings_id_from_tag(self.embeddings_tag)
            object.__setattr__(self, "embeddings_id", resolved_id)

        # Set default if neither embeddings nor embeddings_id nor embeddings_tag provided
        if self.embeddings_id is None:
            default_id = global_config().get_str("embeddings.models.default")
            object.__setattr__(self, "embeddings_id", default_id)

        # Final validation that the resolved/default embeddings_id is known
        if self.embeddings_id not in EmbeddingsFactory.known_items():
            raise ValueError(
                f"Unknown embeddings: {self.embeddings_id}; Check API key and module imports. Should be in {EmbeddingsFactory.known_items()}"
            )

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[EmbeddingsInfo]:
        """List all known embeddings models.

        Returns:
            List of all configured embeddings models
        """
        return _read_embeddings_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, EmbeddingsInfo]:
        """Create a dictionary of available embeddings models.

        Returns:
            Dictionary mapping model IDs to their configurations
        """
        return {
            item.id: item
            for item in EmbeddingsFactory.known_list()
            if get_provider_api_env_var(item.provider) is not None
            and (get_provider_api_env_var(item.provider) == "" or get_provider_api_env_var(item.provider) in os.environ)
        }

    @staticmethod
    def known_items() -> list[str]:
        """List identifiers of available embeddings models.

        Returns:
            List of model identifiers
        """
        return sorted(EmbeddingsFactory.known_items_dict().keys())

    @staticmethod
    def find_embeddings_id_from_tag(embeddings_tag: str) -> str:
        """Find embeddings ID from tag in configuration.

        Args:
            embeddings_tag: Tag to lookup (e.g., 'local', 'default')

        Returns:
            Embeddings ID corresponding to the tag

        Raises:
            ValueError: If tag is not found or corresponds to unknown embeddings
        """
        embeddings_id = global_config().get_str(f"embeddings.models.{embeddings_tag}", default="default")
        if embeddings_id == "default":
            raise ValueError(f"Cannot find embeddings of type: '{embeddings_tag}' (no key found in config file)")
        if embeddings_id not in EmbeddingsFactory.known_items():
            raise ValueError(f"Cannot find embeddings '{embeddings_id}' of type: '{embeddings_tag}'")
        return embeddings_id

    @staticmethod
    def resolve_embeddings_identifier(embeddings: str) -> str:
        """Resolve a unified embeddings identifier to an actual embeddings ID.

        This function accepts a string that could be either an embeddings ID or an embeddings tag
        and returns the corresponding embeddings ID.

        Args:
            embeddings: A string that could be either an embeddings ID or an embeddings tag

        Returns:
            The resolved embeddings ID

        Raises:
            ValueError: If the provided string is neither a valid embeddings ID nor a valid embeddings tag
        """
        # Check if it's a known embeddings ID
        if embeddings in EmbeddingsFactory.known_items():
            return embeddings

        # Otherwise, try to resolve it as a tag
        try:
            return EmbeddingsFactory.find_embeddings_id_from_tag(embeddings)
        except ValueError as ex:
            # If not a tag either, give a helpful error message
            raise ValueError(
                f"Unknown embeddings identifier '{embeddings}'. It is neither a valid embeddings ID nor a valid embeddings tag. "
                f"Valid embeddings IDs: {EmbeddingsFactory.known_items()}"
            ) from ex

    @staticmethod
    def resolve_embeddings_identifier_safe(embeddings: str) -> tuple[str | None, str | None]:
        """Safely resolve a unified embeddings identifier to an actual embeddings ID.

        This function accepts a string that could be either an embeddings ID or an embeddings tag
        and returns the corresponding embeddings ID or an error message.

        Args:
            embeddings: A string that could be either an embeddings ID or an embeddings tag

        Returns:
            A tuple of (resolved_id, error_message). If successful, error_message is None.
            If unsuccessful, resolved_id is None and error_message contains user guidance.
        """
        try:
            resolved_id = EmbeddingsFactory.resolve_embeddings_identifier(embeddings)
            return resolved_id, None
        except ValueError:
            error_msg = (
                f"❌ Unknown embeddings identifier '{embeddings}'.\n\n"
                f"💡 To see available options, try:\n"
                f"   • uv run cli info config    (shows embeddings tags like 'local', 'default')\n"
                f"   • uv run cli info models    (shows all available embeddings IDs)\n\n"
                f"🏷️  Available embeddings tags: Use tags defined in your config for easier access\n"
                f"🆔 Available embeddings IDs: {', '.join(EmbeddingsFactory.known_items()[:3])}{'...' if len(EmbeddingsFactory.known_items()) > 3 else ''}"
            )
            return None, error_msg

    def get(self) -> Embeddings:
        """Create an embeddings model instance."""
        provider = self.info.provider
        env_var = get_provider_api_env_var(provider)
        # Only require API key if provider needs one (env_var is not empty string)
        if env_var is None or (env_var != "" and env_var not in os.environ):
            raise EnvironmentError(f"No known API key for : {self.info.id}")
        embeddings = self.model_factory()
        if self.cache_embeddings:
            embeddings = self.get_cached_embedder(embeddings)
        return embeddings

    def model_factory(self) -> Embeddings:
        """Create an embeddings model based on configuration.

        Returns:
            Instantiated embeddings model

        Raises:
            ValueError: If embeddings model is not supported
        """
        api_key = get_provider_api_key(self.info.provider)

        if self.info.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(api_key=api_key)
        elif self.info.provider == "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore  # noqa: I001

            emb = GoogleGenerativeAIEmbeddings(model=self.info.model, google_api_key=api_key)  # type: ignore
        elif self.info.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

            cache = global_config().get_str("embeddings.cache")
            emb = HuggingFaceEmbeddings(
                model_name=self.info.model,
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder=cache,
            )
        elif self.info.provider == "edenai":
            from langchain_community.embeddings.edenai import EdenAiEmbeddings

            provider, _, model = self.info.model.partition("/")
            emb = EdenAiEmbeddings(model=model, provider=provider, edenai_api_key=api_key)
        elif self.info.provider == "azure":
            from langchain_openai import AzureOpenAIEmbeddings

            name, _, api_version = self.info.model.partition("/")
            emb = AzureOpenAIEmbeddings(
                azure_deployment=name,
                model=name,
                api_version=api_version,
                api_key=api_key or None,
            )
        elif self.info.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(model=self.info.model)
        elif self.info.provider == "deepinfra":
            from langchain_community.embeddings import DeepInfraEmbeddings

            emb = DeepInfraEmbeddings(
                model_id=self.info.model, deepinfra_api_token=api_key.get_secret_value() if api_key else None
            )
        elif self.info.provider == "openrouter":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(
                model=self.info.model,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        elif self.info.provider == "fake":
            from langchain_community.embeddings import DeterministicFakeEmbedding

            emb = DeterministicFakeEmbedding(size=768)  # Default size matching common embedding dimensions
        else:
            raise ValueError(f"unsupported Embeddings class {self.info.provider}")
        return emb

    def get_cached_embedder(self, underlying_embeddings: Embeddings) -> CacheBackedEmbeddings:
        """Create a cached embeddings model.

        Returns:
            Cached embeddings model with persistent storage
        """

        registry = KvStoreRegistry()
        kv_store = registry.get(
            store_id="default", namespace="cache_embeddings"
        )  # TODO : support SQL  (need async KvStore)
        base = f"{self.short_name()}-"
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=kv_store,
            namespace=base,
            key_encoder="sha256",
        )
        return cached_embedder

    def short_name(self) -> str:
        """Return the model ID without the provider (everything before the last underscore)."""
        return self.info.id.rsplit("_", maxsplit=1)[0]

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings model from configuration."""
        if self.info.dimension is None:
            raise ValueError(f"Dimension not configured for model '{self.info.id}'")
        return self.info.dimension


def get_embeddings(
    embeddings: str | None = None,
    embeddings_id: str | None = None,
    embeddings_tag: str | None = None,
    encoding_str: str | None = None,
    retrieving_str: str | None = None,
    cache_embeddings: bool = False,
) -> Embeddings:
    """Retrieve an embeddings model with optional configuration.

    Provides a convenient way to get an embeddings model with flexible
    configuration options.

    Args:
        embeddings: Unified embeddings identifier (can be either ID or tag) - recommended
        embeddings_id: (Deprecated) Unique identifier for the embeddings model
        embeddings_tag: (Deprecated) Tag (type) of embeddings to use (local, default, etc.)
        encoding_str: Optional encoding configuration string
        retrieving_str: Optional retrieving configuration string
        cache_embeddings: Whether to cache embeddings

    Returns:
        Configured embeddings model

    Examples:
        ```python
        # Get default embeddings
        embeddings = get_embeddings()

        # Get specific model by ID (recommended)
        embeddings = get_embeddings(embeddings="ada_002_openai")

        # Get model by tag (recommended)
        embeddings = get_embeddings(embeddings="local")

        # Deprecated ways (still work but will show warnings)
        embeddings = get_embeddings(embeddings_id="ada_002_openai")
        embeddings = get_embeddings(embeddings_tag="local")

        # Use the embeddings
        vectors = embeddings.embed_documents(["Sample text"])
        ```
    """
    # Show deprecation warnings for old parameters
    if embeddings_id is not None:
        logger.warning(
            "⚠️  'embeddings_id' parameter in get_embeddings() is deprecated. Use 'embeddings' instead. "
            "Example: get_embeddings(embeddings='ada_002_openai')"
        )
    if embeddings_tag is not None:
        logger.warning(
            "⚠️  'embeddings_tag' parameter in get_embeddings() is deprecated. Use 'embeddings' instead. "
            "Example: get_embeddings(embeddings='local')"
        )

    factory = EmbeddingsFactory(
        embeddings=embeddings,
        embeddings_id=embeddings_id,
        embeddings_tag=embeddings_tag,
        encoding_str=encoding_str,
        retrieving_str=retrieving_str,
        cache_embeddings=cache_embeddings,
    )
    info = f"get embeddings: '{factory.embeddings_id}'"
    if embeddings_tag:
        info += f" (from tag: '{embeddings_tag}')"
    info += f" -cache: {cache_embeddings}" if cache_embeddings else ""
    logger.debug(info)
    return factory.get()


# QUICK TEST
if __name__ == "__main__":
    embedder = get_embeddings(embeddings="ada_002_openai")
