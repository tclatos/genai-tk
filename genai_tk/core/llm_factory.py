"""Language Model (LLM) factory and configuration management.

This module implements a factory pattern for creating and managing Language Learning Models
from various providers. It handles model configuration, runtime switching, and integration
with caching and streaming features.

Key Features:
- Support for multiple providers (OpenAI, DeepInfra, Groq, etc.)
- Runtime model switching and fallback mechanisms
- Structured JSON output support
- Caching and streaming capabilities
- Configuration through YAML files
- API key management via environment variables

Models are identified by unique IDs following the pattern: model_version@provider
Example: gpt_4o@openai for GPT-4o from OpenAI

Configuration is stored in models_providers.yaml and supports:
- Default model selection
- API key management
- Cache configuration
- Streaming options

Example:
    >>> # Get default LLM
    >>> llm = get_llm()

    >>> # Get specific model with JSON output
    >>> llm = get_llm(llm="gpt_4o@openai", json_mode=True)

    >>> # Get LLM with specific configuration
    >>> llm = get_llm(llm="gpt_4o@openai")
"""


# TODO
#  implement from langchain_core.rate_limiters import InMemoryRateLimiter

import importlib.util
import os
import re
from functools import cached_property, lru_cache
from typing import Annotated, Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from loguru import logger
from omegaconf import DictConfig, ListConfig
from pydantic import BaseModel, Field, SecretStr, computed_field, field_validator

from genai_tk.core.cache import LlmCache
from genai_tk.core.providers import (
    OPENROUTER_API_BASE,
    PROVIDER_INFO,
    get_provider_api_env_var,
    get_provider_api_key,
    get_provider_info,
)
from genai_tk.utils.config_mngr import global_config

SEED = 42  # Arbitrary value....
DEFAULT_MAX_RETRIES = 2


def _is_litellm(text: str) -> bool:
    """
    Validate if a text matches the pattern with 1-2 slashes and length requirements.
    """
    PATTERN = r"^([a-zA-Z0-9_.-]{6,}/){1,2}[a-zA-Z0-9_.-]{11,}$"
    return bool(re.match(PATTERN, text))


class LlmInfo(BaseModel):
    """Description of an LLM model and its configuration.

    Attributes:
        id: Unique identifier in format model_id@provider (e.g. gpt_4o@openai)
        provider: name of the provider
        model: Model identifier used by the provider
        api_key_env_var: API key environment variable name (computed from cls)
        config: Additional configuration for the provider
    """

    # an ID for the LLM; should follow Python variables constraints
    id: str
    provider: str
    model: str  # Name of the model for the constructor
    llm_args: dict[str, Any] = {}

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        # Ensure the ID contains @ separator between model and provider
        if "@" not in v:
            raise ValueError("id must contain @ separator between model and provider (e.g. 'gpt_4o@openai')")
        parts = v.split("@")
        if len(parts) != 2:
            raise ValueError("id must have exactly one @ separator (format: model@provider)")
        return v

    @property
    def llm(self) -> str:
        """Backward compatibility property. Returns the id field."""
        return self.id

    def get_provider_info(self):
        """Get ProviderInfo for this LLM's provider."""
        return get_provider_info(self.provider)

    def get_use_string(self) -> str:
        """Get the 'use' string in Deer-flow format (module:ClassName)."""
        return self.get_provider_info().get_use_string()

    def get_api_base(self) -> str | None:
        """Get the API base URL for this provider if applicable."""
        return self.get_provider_info().api_base

    def get_api_key_env_var(self) -> str:
        """Get the API key environment variable name for this provider."""
        return self.get_provider_info().api_key_env_var


def _read_llm_list_file() -> list[LlmInfo]:
    """Read LLM providers from merged configuration.

    The providers are now merged into the main config via :merge in app_conf.yaml,
    so we read from llm instead of loading a separate file.
    """
    from genai_tk.utils.config_exceptions import ConfigKeyNotFoundError, ConfigTypeError

    try:
        providers_data = global_config().get("llm.registry")
    except Exception as e:
        logger.error(
            "Failed to load LLM providers from config. "
            "Ensure 'config/providers/llm.yaml' is in the :merge list in app_conf.yaml"
        )
        raise ConfigKeyNotFoundError(
            "llm.registry",
            available_keys=[],
        ) from e

    if not isinstance(providers_data, (list, ListConfig)):
        raise ConfigTypeError(
            "llm.registry", expected_type=list, actual_type=type(providers_data), actual_value=providers_data
        )

    llms = []
    for idx, model_entry in enumerate(providers_data):
        if not model_entry or not isinstance(model_entry, (dict, DictConfig)):
            logger.warning(f"Skipping invalid model entry at index {idx}: {model_entry}")
            continue

        # The model entry has this structure:
        # - model_id: xxx
        #   providers: [...]
        model_id = model_entry.get("model_id")

        if not model_id:
            logger.warning(f"Skipping model entry without model_id at index {idx}: {model_entry}")
            continue

        providers_list = model_entry.get("providers", [])
        if not providers_list:
            logger.debug(f"Model {model_id} has no providers")
            continue

        for provider_info in providers_list:
            if isinstance(provider_info, (dict, DictConfig)):
                # provider can be a dict with configuration
                for provider, config in provider_info.items():
                    if isinstance(config, (dict, DictConfig)):
                        # Complex configuration (like vllm or custom)
                        model_name = config.pop("model", "")
                        llm_info = {
                            "id": f"{model_id}@{provider}",
                            "provider": provider,
                            "model": model_name,
                            "llm_args": config,
                        }
                    else:
                        # Simple string configuration
                        llm_info = {
                            "id": f"{model_id}@{provider}",
                            "provider": provider,
                            "model": str(config),
                            "llm_args": {},
                        }
                    llms.append(LlmInfo(**llm_info))
            else:
                logger.warning(f"Unexpected provider format for model {model_id}: {provider_info}")

    if not llms:
        logger.warning("No LLM providers found in configuration")

    return llms


class LlmFactory(BaseModel):
    """Factory for creating and configuring LLM instances.

    Handles the creation of LangChain BaseLanguageModel instances with appropriate
    configuration based on the model type and provider.

    Attributes:
        llm: Unified LLM identifier (can be either LLM ID or tag from config)
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        reasoning: Whether to show reasoning/thinking process (None=default, True=enable, False=disable)
        cache: cache method ("sqlite", "memory", "no_cache", ..) or "default", or None if no change (global setting)
        llm_params: other llm parameters (temperature, max_token, ....)
    """

    llm: Annotated[str | None, Field(validate_default=True)] = None
    json_mode: bool = False
    streaming: bool = False
    reasoning: bool | None = None
    cache: str | None = None
    llm_params: dict = {}

    # Internal field set during resolution
    llm_id: Annotated[str | None, Field(validate_default=True)] = None

    @property
    def provider(self) -> str:
        """Extract provider from the ID (part after @ separator)."""
        return self.info.id.split("@")[1]

    @computed_field
    @cached_property
    def info(self) -> LlmInfo:
        """Return LLM_INFO information on LLM."""
        if self.llm_id:
            return LlmFactory.known_items_dict().get(self.llm_id)  # type: ignore
        elif self.llm is not None and _is_litellm(self.llm):
            # TO BE CONTINUED !!
            return LlmInfo(id=self.llm, provider="litellm", model=self.llm, llm_args={})
        else:
            raise Exception()

    def model_post_init(self, __context: dict) -> None:
        """Post-initialization validation and ID resolution."""
        # Resolve unified 'llm' parameter (can be ID or tag)
        if self.llm is not None:
            resolved_id = LlmFactory.resolve_llm_identifier(self.llm)
            object.__setattr__(self, "llm_id", resolved_id)

        # Set default if llm not provided
        if self.llm_id is None:
            default_id = global_config().get_str("llm.models.default")
            object.__setattr__(self, "llm_id", default_id)

        # Final validation that the resolved/default llm_id is known
        if self.llm_id not in LlmFactory.known_items():
            raise ValueError(
                f"Unknown LLM: {self.llm_id}; Check API key and module imports. Should be in {LlmFactory.known_items()}"
            )

    @field_validator("cache")
    def check_known_cache(cls, cache: str | None) -> str | None:
        if cache and cache not in LlmCache.values():
            raise ValueError(f"Unknown cache method: '{cache} '; Should be in {LlmCache.values()}")
        return cache

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[LlmInfo]:
        return _read_llm_list_file()

    @staticmethod
    def known_items_dict(explain: bool = False) -> dict[str, LlmInfo]:
        """Return known LLM in the registry whose API key environment variable is known and module can be imported.

        If 'explain', add information on debug.trace
        """
        known_items = {}
        for item in LlmFactory.known_list():
            provider_info = PROVIDER_INFO.get(item.provider)
            if not provider_info:
                if explain:
                    logger.debug(f"No PROVIDER_INFO for LLM provider {item.provider}")
                continue

            module_name = provider_info.module
            api_key_env_var = provider_info.api_key_env_var

            if api_key_env_var in os.environ or api_key_env_var == "":
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    known_items[item.id] = item
                elif explain:
                    logger.debug(f"Module {module_name} for {item.provider} could not be imported.")
            elif explain:
                logger.debug(f"No API key {api_key_env_var} for {item.provider}")
        return known_items

    @staticmethod
    def known_items() -> list[str]:
        """Return id of known LLM in the registry whose API key is available and Python module installed."""
        return sorted(LlmFactory.known_items_dict().keys())

    @staticmethod
    def find_llm_id_from_tag(llm_tag: str) -> str:
        llm_id = global_config().get_str(f"llm.models.{llm_tag}", default="default")
        if llm_id == "default":
            raise ValueError(f"Cannot find LLM of type type : '{llm_tag}' (no key found in config file)")
        if llm_id not in LlmFactory.known_items():
            raise ValueError(f"Cannot find LLM '{llm_id}' of type : '{llm_tag}'")
        return llm_id

    @staticmethod
    def resolve_llm_identifier(llm: str) -> str:
        """Resolve a unified LLM identifier to an actual LLM ID.

        This function accepts a string that could be either an LLM ID or an LLM tag
        and returns the corresponding LLM ID.

        Args:
            llm: A string that could be either an LLM ID or an LLM tag

        Returns:
            The resolved LLM ID

        Raises:
            ValueError: If the provided string is neither a valid LLM ID nor a valid LLM tag
        """
        # Check if it's a known LLM ID
        if llm in LlmFactory.known_items():
            return llm

        # Otherwise, try to resolve it as a tag
        try:
            return LlmFactory.find_llm_id_from_tag(llm)
        except ValueError as ex:
            if _is_litellm(llm):
                raise NotImplementedError("Support of LiteLLM model names not et implemented") from ex
            # If not a tag either, give a helpful error message
            raise ValueError(
                f"Unknown LLM identifier '{llm}'. It is neither a valid LLM ID nor a valid LLM tag. "
                f"Valid LLM IDs: {LlmFactory.known_items()}"
            ) from ex

    @staticmethod
    def resolve_llm_identifier_safe(llm: str) -> tuple[str | None, str | None]:
        """Safely resolve a unified LLM identifier to an actual LLM ID.

        This function accepts a string that could be either an LLM ID or an LLM tag
        and returns the corresponding LLM ID or an error message.

        Args:
            llm: A string that could be either an LLM ID or an LLM tag

        Returns:
            A tuple of (resolved_id, error_message). If successful, error_message is None.
            If unsuccessful, resolved_id is None and error_message contains user guidance.
        """
        try:
            resolved_id = LlmFactory.resolve_llm_identifier(llm)
            return resolved_id, None
        except ValueError:
            error_msg = (
                f"❌ Unknown LLM identifier '{llm}'.\n\n"
                f"💡 To see available options, try:\n"
                f"   • uv run cli info config    (shows LLM tags like 'fast_model', 'powerful_model')\n"
                f"   • uv run cli info models    (shows all available LLM IDs)\n\n"
                f"🏷️  Available LLM tags: Use tags defined in your config for easier access\n"
                f"🆔 Available LLM IDs: {', '.join(LlmFactory.known_items()[:3])}"
                f"{'...' if len(LlmFactory.known_items()) > 3 else ''}"
            )
            return None, error_msg

    def get_id(self) -> str:
        """Return the id of the LLM."""
        assert self.llm_id
        return self.llm_id

    def short_name(self) -> str:
        """Return the model ID without the provider (everything before @ separator)."""
        return self.info.id.split("@")[0]

    def get_litellm_model_name(self, separator: str = "/") -> str:
        """Return the LiteLLM id string from our llm_id  (best effort).

        Not all cases covered.
        """
        if self.provider == "openai":
            result = f"{self.info.model}"
        else:
            result = f"{self.provider}/{self.info.model}"
        try:
            get_llm_provider(result)
            # Note: LiteLLM mentions a 'get_valid_models' call, but not seems present

        except Exception as ex:
            raise ValueError(f"Incorrect or unknown LiteLLM provider for: '{result}'") from ex

        # Replace first slash with separator if it exists
        if "/" in result:
            parts = result.split("/", 1)
            return f"{parts[0]}{separator}{parts[1]}"
        return result

    def get_smolagent_model(self):  # -> ApiModel
        from smolagents import AzureOpenAIServerModel, LiteLLMModel

        # Seems better to set these variables, nut not sure.

        if self.provider == "azure":
            name, _, api_version = self.info.model.partition("/")
            model = AzureOpenAIServerModel(
                model_id=name,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
            )
        else:
            model = LiteLLMModel(model_id=self.get_litellm_model_name(), **self.llm_params)
        return model

    def get(self) -> BaseChatModel:
        """Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the known_llm_list() table, otherwise
        we create the LLM from a LangChain LLM class.

        Example:
            ```python
            # Get default LLM
            llm = LlmFactory().get()

            # Get specific model with JSON output
            llm = LlmFactory(llm="gpt_35_openai", json_mode=True).get()

            # Generate a joke
            joke = llm.invoke("Tell me a joke about AI")
            ```
        """
        api_key_env_var = get_provider_api_env_var(self.info.provider)
        if api_key_env_var and not os.getenv(api_key_env_var, "").strip():
            raise EnvironmentError(
                f"API key environment variable '{api_key_env_var}' not set or empty for : {self.llm_id}"
            )
        llm = self.model_factory()
        return llm

    def model_factory(self) -> BaseChatModel:
        """Model factory, according to the model class."""
        from langchain_core.globals import get_llm_cache
        from langchain_openai import ChatOpenAI

        from genai_tk.core.cache import LlmCache

        if self.cache:
            lc_cache = LlmCache.from_value(self.cache)
        else:
            lc_cache = get_llm_cache()
        common_params = {
            "temperature": 0.0,
            "cache": lc_cache,
            "seed": SEED,
            "max_retries": DEFAULT_MAX_RETRIES,
            "streaming": self.streaming,
        }
        api_key = get_provider_api_key(self.info.provider)
        llm_params = common_params | self.llm_params
        if self.json_mode:
            llm_params |= {"response_format": {"type": "json_object"}}

        # Providers that require custom implementation (not supported by init_chat_model or need special handling)
        CUSTOM_IMPLEMENTATION_PROVIDERS = {
            "deepinfra",
            "edenai",
            "google",
            "azure",
            "openrouter",
            "huggingface",
            "litellm",
            "custom",
            "ollama",
            "fake",
        }

        # case for most "standard" providers -> we use LangChain factory
        if self.info.provider not in CUSTOM_IMPLEMENTATION_PROVIDERS:
            # Some parameters are handled differently between provider. Here some workaround:
            if self.info.provider in ["groq", "mistralai"]:
                seed = llm_params.pop("seed")
                if self.info.provider == "groq":
                    llm_params |= {"model_kwargs": {"seed": seed}}
            llm = init_chat_model(
                model=self.info.model, model_provider=self.info.provider, api_key=api_key, **llm_params
            )

        elif self.info.provider == "deepinfra":
            from langchain_community.chat_models.deepinfra import ChatDeepInfra

            llm = ChatDeepInfra(
                name=self.info.model,
                deepinfra_api_token=str(api_key),
                **llm_params,
            )
        elif self.info.provider == "edenai":
            from langchain_community.chat_models.edenai import ChatEdenAI

            provider, _, model = self.info.model.partition("/")
            _ = llm_params.pop("seed")
            _ = llm_params.pop("json_object", None)
            _ = llm_params.pop("max_retries")

            llm = ChatEdenAI(
                provider=provider,
                model=model,
                edenai_api_key=api_key,
                **llm_params,
            )

        elif self.info.provider == "google":
            from langchain_google_vertexai import ChatVertexAI  # type: ignore  # noqa: I001

            llm = ChatVertexAI(
                model=self.info.model,
                project="prj-p-eden",  # TODO : set it in config
                convert_system_message_to_human=True,
                **llm_params,
            )  # type: ignore
            assert not self.json_mode, "json_mode not supported or coded"

        elif self.info.provider == "azure":
            from langchain_openai import AzureChatOpenAI

            name, _, api_version = self.info.model.partition("/")
            llm = AzureChatOpenAI(
                name=name,
                azure_deployment=name,
                model=name,  # Not sure it's needed
                api_version=api_version,
                api_key=api_key,
                **llm_params,
            )
            if self.json_mode:
                llm = cast(BaseLanguageModel, llm.bind(response_format={"type": "json_object"}))
        elif self.info.provider == "openrouter":
            # See https://openrouter.ai/docs/parameters

            # _ = llm_params.pop("response_format", None) or {}
            # Not sure.  See https://openrouter.ai/docs/structured-outputs

            #  Attempt to avoid fp4 quantization.  But might not work for all cases
            openrouter_provider = self.info.model.partition("/")[0]
            extra_body = None
            if openrouter_provider not in ["openai", "anthropic", "mistralai"]:
                extra_body = {"provider": {"quantizations": ["fp8", "unknown", "fp16", "fp32", "bf16"]}}

            llm = ChatOpenAI(
                base_url=OPENROUTER_API_BASE,
                model=self.info.model,
                api_key=api_key,
                extra_body=extra_body,
                **llm_params,
            )
        elif self.info.provider == "huggingface":
            # NOT WELL TESTED
            # Also consider : https://huggingface.co/blog/inference-providers
            # see https://huggingface.co/blog/langchain
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # type: ignore

            llm = HuggingFaceEndpoint(
                repo_id=self.info.model,
                task="text-generation",
                do_sample=False,
            )  # type: ignore
            return ChatHuggingFace(llm=llm)
        elif self.info.provider == "litellm":
            from langchain_litellm import ChatLiteLLM

            llm = ChatLiteLLM(
                model=self.info.model,
                **llm_params,
            )
        elif self.info.provider == "custom":
            # to be used for vLLM and other providers that comply with OpenAI API

            if not self.info.llm_args.get("model"):
                raise ValueError("'custom' provider should have a 'model' key")

            api_key_str = self.info.llm_args.get("api_key") or "dummy-key"
            llm = ChatOpenAI(
                model=self.info.model,
                api_key=SecretStr(api_key_str),
                **self.info.llm_args,
                **llm_params,
            )

        elif self.info.provider == "ollama":
            import os

            from langchain_ollama import ChatOllama

            # Temporarily disable proxy environment variables for localhost connections
            # This is necessary because Ollama runs locally and shouldn't go through corporate proxies
            original_proxy_env = {}
            proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]

            for var in proxy_vars:
                if var in os.environ:
                    original_proxy_env[var] = os.environ[var]
                    del os.environ[var]

            try:
                # Set reasoning parameter based on factory setting
                # Default to False for cleaner output unless explicitly enabled
                reasoning_enabled = self.reasoning if self.reasoning is not None else False

                llm = ChatOllama(
                    model=self.info.model,
                    base_url="http://localhost:11434",
                    reasoning=reasoning_enabled,
                    **llm_params,
                )
            finally:
                # Restore original proxy settings
                for var, value in original_proxy_env.items():
                    os.environ[var] = value
        elif self.info.provider == "fake":
            from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel

            if self.info.model == "parrot":
                llm = ParrotFakeChatModel()
            else:
                raise ValueError(f"unsupported fake model {self.info.model}")

        else:
            if self.info.provider in LlmFactory.known_items():
                raise EnvironmentError(f"No API key found for LLM: {self.info.provider}")
            else:
                raise ValueError(f"unsupported LLM class {self.info.provider}")

        return llm  # type: ignore


def get_llm(
    llm: str | None = None,
    llm_id: str | None = None,
    llm_tag: str | None = None,
    json_mode: bool = False,
    streaming: bool = False,
    reasoning: bool | None = None,
    cache: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create a configured LangChain BaseLanguageModel instance.

    Args:
        llm: Unified LLM identifier (can be either LLM ID or tag) - recommended
        llm_id: (Deprecated) Unique model identifier (if None, uses default from config)
        llm_tag: (Deprecated) Tag (type) of model to use (fast_model, smart_model, etc.)
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        reasoning: Whether to show reasoning/thinking process (None=default, True=enable, False=disable)
        cache: cache method ("sqlite", "memory", no"_cache, ..) or "default", or None if no change (global setting)
        **kwargs: other llm parameters (temperature, max_token, ....)

    Returns:
        BaseLanguageModel: Configured language model instance

    Examples:
        ```python
        # Get default LLM
        llm = get_llm()

        # Get specific model with streaming (recommended)
        llm = get_llm(llm="gpt_35_openai", streaming=True)

        # Get model by tag (recommended)
        llm = get_llm(llm="fast_model", temperature=0.7)

        # Deprecated ways (still work but will show warnings)
        llm = get_llm(llm_id="gpt_35_openai", streaming=True)
        llm = get_llm(llm_tag="fast_model", temperature=0.7)

        # Use in a chain
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        chain = prompt | get_llm(llm="gpt_4o_openai")
        result = chain.invoke({"topic": "AI"})
        ```
    """
    # Show deprecation warnings for old parameters
    if llm_id is not None:
        logger.warning(
            "⚠️  'llm_id' parameter in get_llm() is deprecated. Use 'llm' instead. Example: get_llm(llm='gpt_35_openai')"
        )
    if llm_tag is not None:
        logger.warning(
            "⚠️  'llm_tag' parameter in get_llm() is deprecated. Use 'llm' instead. Example: get_llm(llm='fast_model')"
        )

    # Handle deprecated llm_id and llm_tag by converting to unified llm parameter
    resolved_llm = llm
    if llm is None and llm_id is not None:
        resolved_llm = llm_id
    elif llm is None and llm_tag is not None:
        resolved_llm = llm_tag

    factory = LlmFactory(
        llm=resolved_llm,
        json_mode=json_mode,
        streaming=streaming,
        reasoning=reasoning,
        cache=cache,
        llm_params=kwargs,
    )
    info = f"get LLM:'{factory.llm_id}'"
    info += " -streaming" if streaming else ""
    info += " -json_mode" if json_mode else ""
    info += f" -reasoning: {reasoning}" if reasoning is not None else ""
    info += f" -cache: {cache}" if cache else ""
    info += f" -extra: {kwargs}" if kwargs else ""
    logger.debug(info)
    return factory.get()


def get_llm_unified(
    llm: str | None = None,
    json_mode: bool = False,
    streaming: bool = False,
    reasoning: bool | None = None,
    cache: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create a configured LangChain BaseLanguageModel instance using unified LLM parameter.

    Args:
        llm: Unified LLM identifier (can be either LLM ID or LLM tag)
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        reasoning: Whether to show reasoning/thinking process (None=default, True=enable, False=disable)
        cache: cache method ("sqlite", "memory", no"_cache, ..) or "default", or None if no change (global setting)
        **kwargs: other llm parameters (temperature, max_token, ....)

    Returns:
        BaseLanguageModel: Configured language model instance

    Examples:
        ```python
        # Get default LLM
        llm = get_llm_unified()

        # Get specific model by ID
        llm = get_llm_unified(llm="gpt_35_openai", streaming=True)

        # Get model by tag
        llm = get_llm_unified(llm="fast_model", temperature=0.7)
        ```
    """
    factory = LlmFactory(
        llm=llm,
        json_mode=json_mode,
        streaming=streaming,
        reasoning=reasoning,
        cache=cache,
        llm_params=kwargs,
    )
    info = f"get LLM (unified):'{factory.llm_id}'"
    info += " -streaming" if streaming else ""
    info += " -json_mode" if json_mode else ""
    info += f" -reasoning: {reasoning}" if reasoning is not None else ""
    info += f" -cache: {cache}" if cache else ""
    info += f" -extra: {kwargs}" if kwargs else ""
    logger.debug(info)
    return factory.get()


def get_llm_info(llm_id: str) -> LlmInfo:
    """Return information on given LLM."""
    factory = LlmFactory(llm=llm_id)
    r = factory.known_items_dict().get(llm_id)
    if r is None:
        raise ValueError(f"Unknown llm_id: '{llm_id}' ")
    else:
        return r


def llm_config(llm_id: str) -> RunnableConfig:
    """Return a 'RunnableConfig' to configure an LLM at run-time. Check LLM is known.

    Examples :
    ```
        r = chain.with_config(llm_config("claude_haiku45_openrouter")).invoke({})
        # or:
        r = graph.invoke({}, config=llm_config("gpt_35_openai") | {"recursion_limit": 6}) )
    ```
    """
    if llm_id not in LlmFactory.known_items():
        raise ValueError(
            f"Unknown LLM: {llm_id}; Check API key and module imports. Should be in {LlmFactory.known_items()}"
        )
    return configurable({"llm_id": llm_id})


def configurable(conf: dict) -> RunnableConfig:
    """Return a dict with key 'configurable', to be used in 'with_config'.

    Example:
    ```
        llm.with_config(configurable({"my_conf": "my_conf_value"})  )

    """
    return {"configurable": conf}


def get_print_chain(string: str = "") -> RunnableLambda:
    """Return a chain that print the passed input and the config. Useful for debugging.

    Example:
    ```
        from genai_tk.core.llm_factory import configurable, get_print_chain

        add_1 = get_print_chain("before") | RunnableLambda(lambda x: x + 1) | get_print_chain("after")
        chain = add_1.with_config(configurable({"my_conf": "my_conf_value"}))
        print(chain.invoke(1))
    ```
    """
    from langchain_core.runnables import RunnableConfig, RunnableLambda

    def fn(input: Any, config: RunnableConfig) -> Any:
        print(string, input, config)
        return input

    return RunnableLambda(fn)


# QUICK TEST
if __name__ == "__main__":
    get_llm("gpt_4o@azure")
