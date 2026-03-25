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

from __future__ import annotations

# TODO
#  implement from langchain_core.rate_limiters import InMemoryRateLimiter
import difflib
import importlib.util
import os
import re
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Annotated, Any, cast

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig, RunnableLambda

from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, computed_field, field_validator

from genai_tk.core.cache import CacheMethod, LlmCache
from genai_tk.core.models_db import ModelEntry, get_models_db
from genai_tk.core.providers import (
    OPENROUTER_API_BASE,
    PROVIDER_INFO,
    ProviderInfo,
    get_provider_api_env_var,
    get_provider_api_key,
    get_provider_info,
)
from genai_tk.utils.config_mngr import global_config

SEED = 42  # Arbitrary value....
DEFAULT_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# LLM configuration schema and typed accessor
# ---------------------------------------------------------------------------


class LlmModelsConfig(BaseModel):
    """Named LLM model IDs for each configured tag (``llm.models:`` in YAML).

    The ``default`` tag is mandatory; any extra tags (e.g. ``fake``, ``fast_model``) are
    stored as extra fields and accessible via ``get_tag()``.
    """

    default: str = Field(..., description="Default LLM model (ID or compact alias)")
    model_config = ConfigDict(extra="allow")

    def get_tag(self, tag: str) -> str | None:
        """Return the LLM ID configured for a named tag (e.g. 'fake', 'fast_model')."""
        if tag == "default":
            return self.default
        return (self.model_extra or {}).get(tag)

    def all_tags(self) -> dict[str, str]:
        """Return all configured tag → LLM ID mappings."""
        result: dict[str, str] = {"default": self.default}
        result.update({k: v for k, v in (self.model_extra or {}).items() if isinstance(v, str)})
        return result


class LlmSection(BaseModel):
    """Top-level ``llm:`` configuration section (baseline.yaml + providers/llm.yaml).

    Example YAML:
    ```yaml
    llm:
      models:
        default: gpt_oss120@openrouter
        fake: parrot_local@fake
      cache: sqlite
      cache_path: data/llm_cache/langchain.db
      exceptions:
        - model_id: parrot_local
          providers:
            - fake: parrot
    ```
    """

    models: LlmModelsConfig = Field(..., description="Default and tagged LLM model IDs")
    cache: str | CacheMethod = Field(
        "no_cache", description="Cache strategy: 'memory' | 'sqlite' | 'no_cache' or CacheMethod enum"
    )
    cache_path: str | None = Field(None, description="SQLite cache file path (required when cache='sqlite')")
    exceptions: list[Any] = Field(default_factory=list, description="Per-model provider overrides from llm.yaml")
    registry: list[Any] | None = Field(None, description="Legacy alias for 'exceptions' – prefer 'exceptions'")
    model_config = ConfigDict(extra="allow")

    @property
    def exception_entries(self) -> list[Any]:
        """Return the provider override list (falls back to legacy 'registry' key)."""
        return self.exceptions or self.registry or []


def _normalize(s: str) -> str:
    """Strip non-alphanumeric characters and lowercase for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fuzzy_match(query: str, candidates: list[str], n: int = 5, cutoff: float = 0.35) -> list[tuple[str, float]]:
    """Return top-n (candidate, score) pairs sorted by descending similarity score."""
    norm_query = _normalize(query)
    scored: list[tuple[str, float]] = []
    for c in candidates:
        score = difflib.SequenceMatcher(None, norm_query, _normalize(c)).ratio()
        if score >= cutoff:
            scored.append((c, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def resolve_model(compact_alias: str, provider_id: str) -> "tuple[str, ModelEntry | None, list[tuple[str, float]]]":
    """Fuzzy-resolve a compact alias to a canonical model name using the models.dev database.

    For direct providers (e.g. openai, anthropic) searches that provider's model list.
    For gateway providers (e.g. openrouter, github) searches openrouter's model catalogue
    which contains vendor-prefixed names like ``openai/gpt-4.1-mini``.

    Args:
        compact_alias: User-supplied alias (e.g. ``haiku45``, ``gpt41mini``).
        provider_id: Provider key from PROVIDER_INFO (e.g. ``anthropic``, ``openrouter``).

    Returns:
        Tuple of (canonical_model_name, entry_or_none, top_matches_with_scores).

    Example:
        ```python
        name, entry, alts = resolve_model("haiku45", "anthropic")
        # name == "claude-haiku-4-5"
        ```
    """
    db = get_models_db()
    provider_info = PROVIDER_INFO.get(provider_id)

    if provider_info and provider_info.gateway:
        # Use openrouter catalogue which covers most vendors in vendor/model format.
        models = db.provider_models("openrouter")
        candidates = list(models.keys())
        matches = _fuzzy_match(compact_alias, candidates)
        if matches:
            best_name, _ = matches[0]
            return best_name, models.get(best_name), matches
        top3 = [c for c, _ in _fuzzy_match(compact_alias, candidates, n=3, cutoff=0)][:3]
        raise ValueError(f"No model matching '{compact_alias}' for gateway provider '{provider_id}'. Closest: {top3}")

    elif provider_info:
        models = db.provider_models(provider_id)
        if not models:
            # Provider not in models.dev — fall back to searching openrouter with vendor prefix
            models = {k: v for k, v in db.provider_models("openrouter").items() if k.startswith(f"{provider_id}/")}
        candidates = list(models.keys())
        if not candidates:
            raise ValueError(
                f"Provider '{provider_id}' has no entries in the models.dev database. "
                "Add the model explicitly to llm.yaml as an exception."
            )
        matches = _fuzzy_match(compact_alias, candidates)
        if matches:
            best_name, _ = matches[0]
            return best_name, models.get(best_name), matches
        top3 = [c for c, _ in _fuzzy_match(compact_alias, candidates, n=3, cutoff=0)][:3]
        raise ValueError(f"No model matching '{compact_alias}' for provider '{provider_id}'. Closest: {top3}")

    else:
        raise ValueError(f"Unknown provider '{provider_id}'.")


def lookup_model_entry(model_name: str, provider: str) -> ModelEntry | None:
    """Look up a model entry from the local models.dev database.

    Supports direct providers (openai, anthropic, mistralai, groq, google) and gateway
    providers (openrouter, github, litellm) that use vendor-prefixed model names such as
    ``openai/gpt-4.1-mini`` or ``anthropic/claude-sonnet-4-5``.

    Version modifiers like ``:exacto`` or ``:free`` are stripped before lookup.

    Example:
        ```python
        e = lookup_model_entry("gpt-4o-mini", "openai")
        e = lookup_model_entry("openai/gpt-4.1-mini", "openrouter")
        ```
    """
    return get_models_db().lookup(provider, model_name)


# Backward-compatible alias
lookup_lc_profile = lookup_model_entry


def profile_to_capabilities(entry: ModelEntry) -> list[str]:
    """Return the capabilities list for a ``ModelEntry``.

    Thin wrapper around ``entry.capabilities`` kept for call-site compatibility.

    Example:
        ```python
        caps = profile_to_capabilities(lookup_model_entry("gpt-4o", "openai"))
        ```
    """
    return entry.capabilities


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
        llm_args: Additional kwargs forwarded to the LLM constructor
        capabilities: Capability override list. When empty, derived from ModelProfile.
        max_tokens: Maximum output tokens override. When None, derived from ModelProfile.
        context_window: Context window override. When None, derived from ModelProfile.
    """

    # an ID for the LLM; should follow Python variables constraints
    id: str
    provider: str
    model: str  # Name of the model for the constructor
    llm_args: dict[str, Any] = {}
    capabilities: list[str] = []
    max_tokens: int | None = None
    context_window: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    @cached_property
    def profile(self) -> ModelEntry | None:
        """Model entry from the local models.dev database, if available for this model."""
        return lookup_model_entry(self.model, self.provider)

    @property
    def effective_capabilities(self) -> list[str]:
        """Capabilities list from YAML override, or derived from the models.dev database."""
        if self.capabilities:
            return self.capabilities
        p = self.profile
        return p.capabilities if p else []

    @property
    def effective_max_tokens(self) -> int | None:
        """Max output tokens from YAML override, or from the models.dev database."""
        if self.max_tokens is not None:
            return self.max_tokens
        p = self.profile
        return p.output if p else None

    @property
    def effective_context_window(self) -> int | None:
        """Input context window from YAML override, or from the models.dev database."""
        if self.context_window is not None:
            return self.context_window
        p = self.profile
        return p.context if p else None

    @property
    def supports_thinking(self) -> bool:
        """True if the model supports explicit thinking/reasoning mode."""
        return "thinking" in self.effective_capabilities

    @property
    def supports_vision(self) -> bool:
        """True if the model supports image/vision inputs."""
        return "vision" in self.effective_capabilities

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


def _llm_section() -> LlmSection:
    """Return typed LLM configuration (the ``llm:`` section of app config).

    Parses ``llm.models``, ``llm.cache``, ``llm.cache_path``, and
    ``llm.exceptions`` into a validated ``LlmSection`` model.

    Example:
        ```python
        from genai_tk.core.llm_factory import _llm_section

        default_model = _llm_section().models.default
        all_tags = _llm_section().models.all_tags()
        ```
    """
    from genai_tk.utils.config_exceptions import yaml_config_validation

    with yaml_config_validation(context="llm"):
        raw = global_config().get_dict("llm")
        return LlmSection.model_validate(raw)


def _read_llm_list_file() -> list[LlmInfo]:
    """Read LLM providers from merged configuration.

    The providers are now merged into the main config via :merge in app_conf.yaml,
    so we read from llm instead of loading a separate file.
    """
    providers_data = _llm_section().exception_entries

    if not providers_data:
        logger.warning("No LLM exceptions found in config. Ensure 'config/providers/llm.yaml' is in the :merge list.")
        return []

    llms = []
    for idx, model_entry in enumerate(providers_data):
        if not model_entry or not isinstance(model_entry, dict):
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

        # Extract model-level metadata (shared across all providers for this model)
        capabilities = list(model_entry.get("capabilities", []) or [])
        max_tokens = model_entry.get("max_tokens", None)
        context_window = model_entry.get("context_window", None)

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
                            "capabilities": capabilities,
                            "max_tokens": max_tokens,
                            "context_window": context_window,
                        }
                    else:
                        # Simple string configuration
                        llm_info = {
                            "id": f"{model_id}@{provider}",
                            "provider": provider,
                            "model": str(config),
                            "llm_args": {},
                            "capabilities": capabilities,
                            "max_tokens": max_tokens,
                            "context_window": context_window,
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

    llm: Annotated[str, Field(validate_default=True)] = "default"
    json_mode: bool = False
    streaming: bool = False
    reasoning: bool | None = None
    cache: str | CacheMethod | None = None
    llm_params: dict = {}

    # Internal fields set during resolution
    llm_id: Annotated[str | None, Field(validate_default=True)] = None
    _resolved_llm_info: LlmInfo | None = PrivateAttr(default=None)

    @field_validator("cache")
    @classmethod
    def validate_cache(cls, v: str | CacheMethod | None) -> str | CacheMethod | None:
        """Validate cache method value.

        Args:
            v: Cache method as string, CacheMethod enum, or None

        Returns:
            Validated cache value

        Raises:
            ValueError: If cache value is invalid
        """
        if v is None:
            return v

        # If it's already a CacheMethod enum, it's valid
        if isinstance(v, CacheMethod):
            return v

        # If it's a string, validate it
        if isinstance(v, str):
            valid_values = ["memory", "sqlite", "no_cache"]
            if v not in valid_values:
                raise ValueError("Input should be 'memory', 'sqlite' or 'no_cache'")
            return v

        raise ValueError(f"cache must be string or CacheMethod, got {type(v)}")

    @property
    def provider(self) -> str:
        """Extract provider from the ID (part after @ separator)."""
        return self.info.id.split("@")[1]

    @computed_field
    @cached_property
    def info(self) -> LlmInfo:
        """Return LLM_INFO information on LLM."""
        if self._resolved_llm_info is not None:
            return self._resolved_llm_info
        if self.llm_id:
            return LlmFactory.known_items_dict().get(self.llm_id)  # type: ignore
        elif self.llm is not None and _is_litellm(self.llm):
            # TO BE CONTINUED !!
            return LlmInfo(id=self.llm, provider="litellm", model=self.llm, llm_args={})
        else:
            raise Exception()

    def model_post_init(self, __context: dict) -> None:
        """Post-initialization validation and ID resolution."""
        # Seed llm_id from the unified 'llm' parameter.
        # "standard Pydantic pattern for setting fields internally during model_post_init when you want to mutate state without re-triggering validation.""
        if self.llm_id is None:
            object.__setattr__(self, "llm_id", self.llm)

        # Resolve llm_id to a canonical form if not already a known item.
        # This handles compact aliases like 'gpt_41mini@openai', config tags,
        # and default config values that may use compact notation.
        if self.llm_id not in LlmFactory.known_items():
            assert self.llm_id is not None  # for type checker
            try:
                resolved_id = LlmFactory.resolve_llm_identifier(self.llm_id)
                object.__setattr__(self, "llm_id", resolved_id)
                # If still not in known_items (e.g. no API key / module), build LlmInfo on-the-fly
                if resolved_id not in LlmFactory.known_items():
                    compact, _, provider_id = resolved_id.rpartition("@")
                    self._resolved_llm_info = LlmInfo(id=resolved_id, provider=provider_id, model=compact)
            except ValueError:
                # Fallback: direct fuzzy resolve for compact@provider aliases
                if "@" in self.llm_id:
                    compact, _, provider_id = self.llm_id.rpartition("@")
                    canon, _profile, _alts = resolve_model(compact, provider_id)
                    self._resolved_llm_info = LlmInfo(id=self.llm_id, provider=provider_id, model=canon)
                else:
                    raise

        # Final validation: reject only if we have no resolved info and no known item
        if self._resolved_llm_info is None and self.llm_id not in LlmFactory.known_items():
            raise ValueError(
                f"Unknown LLM: {self.llm_id}; Check API key and module imports. Should be in {LlmFactory.known_items()}"
            )

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[LlmInfo]:
        return _read_llm_list_file()

    @staticmethod
    def registry_items_dict() -> dict[str, LlmInfo]:
        """Enumerate models from the local models.dev database as LlmInfo objects.

        Returns items like ``claude-haiku-4-5@anthropic`` for each model in each direct
        (non-gateway) provider whose API key is available and Python module is installed.
        Gateway providers (openrouter, litellm, github) are excluded to avoid enumerating
        thousands of cross-vendor aliases.
        """
        db = get_models_db()
        items: dict[str, LlmInfo] = {}
        for provider_name, provider_info in PROVIDER_INFO.items():
            if provider_info.gateway:
                continue  # skip gateway providers
            if provider_info.api_key_env_var and provider_info.api_key_env_var not in os.environ:
                continue
            spec = importlib.util.find_spec(provider_info.module)
            if spec is None:
                continue
            for model_id in db.provider_models(provider_name):
                item_id = f"{model_id}@{provider_name}"
                if item_id not in items:
                    items[item_id] = LlmInfo(id=item_id, provider=provider_name, model=model_id)
        return items

    @staticmethod
    def known_items_dict(explain: bool = False) -> dict[str, LlmInfo]:
        """Return known LLM items: exceptions from llm.yaml merged with models.dev registry entries.

        Items from llm.yaml (exceptions) take precedence over registry entries with the same ID.
        Only providers whose API key is set and whose Python module is importable are included.
        """

        def _provider_available(provider_info: ProviderInfo) -> bool:
            has_key = not provider_info.api_key_env_var or provider_info.api_key_env_var in os.environ
            has_module = importlib.util.find_spec(provider_info.module) is not None
            return has_key and has_module

        # llm.yaml entries (exceptions / overrides)
        yaml_items: dict[str, LlmInfo] = {}
        for item in LlmFactory.known_list():
            provider_info = PROVIDER_INFO.get(item.provider)
            if not provider_info:
                if explain:
                    logger.debug(f"No PROVIDER_INFO for LLM provider {item.provider}")
                continue
            if _provider_available(provider_info):
                yaml_items[item.id] = item
            elif explain:
                logger.debug(f"Skipping {item.id}: missing key or module for {item.provider}")

        # models.dev registry entries (yaml takes precedence)
        registry_items = {k: v for k, v in LlmFactory.registry_items_dict().items() if k not in yaml_items}
        return yaml_items | registry_items

    @staticmethod
    def known_items() -> list[str]:
        """Return canonical IDs of all usable LLMs (API key set + Python module installed).

        Each ID has the form ``canonical-model-name@provider``, e.g. ``gpt-4.1-mini@openai``.
        Sources (in priority order): llm.yaml exceptions, then the models.dev registry.

        LLM identifiers accepted anywhere in the codebase
        ---------------------------------------------------
        1. Canonical ID        – exact entry from this list, e.g. ``gpt-4.1-mini@openai``.
        2. llm.yaml model_id   – underscore alias defined in llm.yaml, e.g. ``gpt_4o_openai``.
        3. Config tag          – logical role from config, e.g. ``fast_model``, ``smart_model``.
        4. Compact alias       – ``<alias>@<provider>`` fuzzy-resolved via models.dev,
                                 e.g. ``gpt41mini@openai`` or ``haiku45@anthropic``.

        Resolution is performed by ``resolve_llm_identifier()`` in that order.
        """
        return sorted(LlmFactory.known_items_dict().keys())

    @staticmethod
    def find_llm_id_from_tag(llm_tag: str) -> str:
        models = _llm_section().models
        llm_id = models.get_tag(llm_tag) or "default"
        if llm_id == "default":
            raise ValueError(f"Cannot find LLM of type type : '{llm_tag}' (no key found in config file)")
        # Return the resolved ID even if it's a gateway model (not in known_items).
        # Downstream resolution in model_post_init handles gateway/alias IDs.
        return llm_id

    @staticmethod
    def resolve_llm_identifier(llm: str) -> str:
        """Resolve a unified LLM identifier to an actual LLM ID.

        Accepts an exact LLM ID, a config tag, a compact alias with provider
        (``alias@provider``), or a hyphenated variant of an llm.yaml model_id
        (e.g. ``gpt-oss120@openrouter`` resolves to ``gpt_oss120@openrouter``).

        For compact aliases, fuzzy-matches against the models.dev database and returns
        the canonical ``model_name@provider`` ID (e.g. ``haiku45@anthropic`` →
        ``claude-haiku-4-5@anthropic``).

        Args:
            llm: An LLM ID, config tag, or compact ``alias@provider`` string.

        Returns:
            The resolved LLM ID.

        Raises:
            ValueError: If the string cannot be resolved to any known LLM.
        """
        # Exact match
        if llm in LlmFactory.known_items():
            return llm

        # Config tag lookup
        try:
            tag_value = LlmFactory.find_llm_id_from_tag(llm)
            # The tag value is often a compact alias (e.g. "gpt_oss120@openrouter");
            # recursively resolve it so callers always get a canonical model name.
            if tag_value != llm:
                try:
                    return LlmFactory.resolve_llm_identifier(tag_value)
                except (ValueError, NotImplementedError):
                    pass
            return tag_value
        except ValueError:
            pass

        if _is_litellm(llm):
            raise NotImplementedError("Support of LiteLLM model names not yet implemented")

        if "@" in llm:
            compact, _, provider_id = llm.rpartition("@")

            # Try normalizing hyphens → underscores (llm.yaml model_ids use underscores)
            normalized = f"{compact.replace('-', '_')}@{provider_id}"
            if normalized in LlmFactory.known_items():
                return normalized

            # Fuzzy resolve via models.dev and return canonical model_id@provider
            try:
                canon, _entry, alts = resolve_model(compact, provider_id)
                canonical_id = f"{canon}@{provider_id}"
                best_score = alts[0][1] if alts else 0.0
                if best_score < 0.6:
                    logger.warning(
                        f"Low-confidence LLM resolution: '{llm}' → '{canonical_id}' "
                        f"(score {best_score:.2f}). Did you mean one of: "
                        f"{[name for name, _ in alts[:3]]}?"
                    )
                if canonical_id in LlmFactory.known_items():
                    return canonical_id
                # Model resolved but not in known_items (no API key / module): still usable
                return canonical_id
            except ValueError:
                pass

        # Build a helpful error: show fuzzy matches from known_items + available config tags
        close_ids = [m for m, _ in _fuzzy_match(llm, LlmFactory.known_items(), n=5, cutoff=0.3)]
        try:
            tags = _llm_section().models.all_tags()
            tags.pop("default", None)
            tags_hint = f"\n  Config tags : {tags}" if tags else ""
        except Exception:
            tags_hint = ""
        suggestions = f"\n  Closest IDs : {close_ids}" if close_ids else ""
        raise ValueError(
            f"Unknown LLM: '{llm}'."
            f"{suggestions}"
            f"{tags_hint}"
            f"\n  Tip: use 'cli info models' to list all IDs, 'cli info config' for tags."
        )

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
        prefix = PROVIDER_INFO[self.provider].litellm_prefix
        if prefix:
            result = f"{prefix}/{self.info.model}"
        else:
            result = self.info.model
        try:
            from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

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
            "github",
            "huggingface",
            "litellm",
            "custom",
            "ollama",
            "fake",
            "mistral",
        }

        # case for most "standard" providers -> we use LangChain factory
        if self.info.provider not in CUSTOM_IMPLEMENTATION_PROVIDERS:
            from langchain.chat_models import init_chat_model

            # Some parameters are handled differently between provider. Here some workaround:
            if self.info.provider == "groq":
                seed = llm_params.pop("seed")
                llm_params |= {"model_kwargs": {"seed": seed}}
            llm = init_chat_model(
                model=self.info.model, model_provider=self.info.provider, api_key=api_key, **llm_params
            )

        elif self.info.provider == "mistral":
            from langchain_mistralai import ChatMistralAI

            llm_params.pop("seed", None)  # MistralAI does not accept seed
            llm = ChatMistralAI(
                model=self.info.model,
                api_key=api_key,
                **llm_params,
            )

        elif self.info.provider == "deepinfra":
            # Use ChatOpenAI with DeepInfra's OpenAI-compatible endpoint.
            # ChatDeepInfra (langchain_community) encodes the model name in the URL path
            # (/v1/inference/{model}), which breaks vendor-prefixed names like
            # "MiniMaxAI/MiniMax-M2.1" (the slash becomes a path separator → 403).
            llm = ChatOpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                model=self.info.model,
                api_key=api_key,
                **llm_params,
            )
        elif self.info.provider == "edenai":
            # EdenAI v3: OpenAI-compatible endpoint — model string is "provider/model"
            # e.g. "openai/gpt-4.1-mini-2025-04-14", "anthropic/claude-sonnet-4-5"
            llm = ChatOpenAI(
                base_url="https://api.edenai.run/v3/llm",
                model=self.info.model,
                api_key=api_key,
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
            # When model has no "/" the api_version is ""; fall back to env vars.
            # AzureChatOpenAI reads OPENAI_API_VERSION natively; many setups use
            # AZURE_OPENAI_API_VERSION instead, so we map it explicitly.
            if not api_version:
                api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("OPENAI_API_VERSION") or ""
            llm = AzureChatOpenAI(
                name=name,
                azure_deployment=name,
                model=name,
                api_version=api_version,
                api_key=api_key,
                **llm_params,
            )
            if self.json_mode:
                from langchain_core.language_models.base import BaseLanguageModel

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
        elif self.info.provider == "github":
            # GitHub Models API - See https://github.com/marketplace/models
            llm = ChatOpenAI(
                base_url="https://models.github.ai/inference",
                model=self.info.model,
                api_key=api_key,
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
    llm: str = "default",
    llm_id: str | None = None,
    llm_tag: str | None = None,
    json_mode: bool = False,
    streaming: bool = False,
    reasoning: bool | None = None,
    cache: str | CacheMethod | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create a configured LangChain BaseLanguageModel instance.

    Args:
        llm: Unified LLM identifier (can be either LLM ID or tag) - recommended
        llm_id: (Deprecated) Unique model identifier
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
    if llm == "default" and llm_id is not None:
        resolved_llm = llm_id
    elif llm == "default" and llm_tag is not None:
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
        # or with compact alias:
        r = chain.with_config(llm_config("gpt_41mini@openai")).invoke({})
    ```
    """
    resolved_id = LlmFactory.resolve_llm_identifier(llm_id)
    return configurable({"llm_id": resolved_id})


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
    from langchain_core.runnables import RunnableLambda

    def fn(input: Any, config: RunnableConfig) -> Any:
        print(string, input, config)
        return input

    return RunnableLambda(fn)


# QUICK TEST
if __name__ == "__main__":
    get_llm("gpt_4o@azure")
