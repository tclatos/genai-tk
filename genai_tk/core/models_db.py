"""Local cache and access layer for the models.dev database.

Downloads and caches the models.dev/api.json database locally and provides fast
lookup of model metadata — capabilities, context limits, pricing — without relying
on any third-party LangChain provider packages.

The database is stored in ``data/models_dev.json`` relative to the workspace root.
Use ``cli info llm-profile <model_id> --reload`` to refresh the database before looking up a model.

Example:
    ```python
    db = get_models_db()
    entry = db.lookup("openai", "gpt-4o-mini")
    if entry:
        print(entry.capabilities)
    ```
"""

import json
import os
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel, field_validator

from genai_tk.utils.singleton import once

MODELS_DEV_URL = "https://models.dev/api.json"
EDENAI_MODELS_URL = "https://api.edenai.run/v3/models"
EDENAI_EU_MODELS_URL = "https://api.eu.edenai.run/v3/models"
EDENAI_API_KEY_ENV_VAR = "EDENAI_API_KEY"
EDENAI_CACHE_FILENAME = "edenai_models.json"
EDENAI_EU_CACHE_FILENAME = "edenai_eur_models.json"
EDENAI_CATALOGUES = {
    "edenai": (EDENAI_MODELS_URL, EDENAI_CACHE_FILENAME),
    "edenai-eur": (EDENAI_EU_MODELS_URL, EDENAI_EU_CACHE_FILENAME),
}


def _default_cache_path() -> Path:
    """Resolve the default models cache path via config, fallback to CWD/data/."""
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().get_dir_path("paths.data") / "models_dev.json"
    except Exception:
        return Path.cwd() / "data" / "models_dev.json"


_DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "models_dev.json"


class ModelRegion(BaseModel):
    """Inference region where a model is available."""

    code: str
    name: str

    model_config = {"frozen": True}


class ModelEntry(BaseModel):
    """Normalized model entry from the models.dev database.

    Attributes:
        id: Model identifier as used by the provider (e.g. ``gpt-4o-mini``)
        name: Human-readable model name
        provider_id: Provider key this entry was loaded from (e.g. ``openai``, ``openrouter``)
        family: Optional model family tag (e.g. ``gpt``, ``claude-sonnet``)
        attachment: True if the model accepts file/image attachments
        reasoning: True if the model supports explicit chain-of-thought / thinking
        tool_call: True if the model supports function/tool calling
        structured_output: True if the model supports structured JSON output natively
        temperature: True if the model accepts a temperature parameter
        modalities_input: List of accepted input modalities (``text``, ``image``, ``audio``, ``video``, ``pdf``)
        modalities_output: List of produced output modalities
        context: Context window size in tokens (``limit.context`` from models.dev)
        output: Max output tokens (``limit.output`` from models.dev)
        cost_input: Cost per million input tokens in USD
        cost_output: Cost per million output tokens in USD
        release_date: ISO date string of initial release
        last_updated: ISO date string of last update
        open_weights: True if model weights are publicly available
        regions: Inference regions where the model is available
    """

    id: str
    name: str
    provider_id: str
    family: str | None = None
    attachment: bool = False
    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    temperature: bool = True
    modalities_input: list[str] = []
    modalities_output: list[str] = []
    context: int | None = None
    output: int | None = None
    cost_input: float | None = None
    cost_output: float | None = None
    release_date: str | None = None
    last_updated: str | None = None
    open_weights: bool = False
    regions: list[ModelRegion] = []

    model_config = {"frozen": True}

    @field_validator("context", "output", mode="before")
    @classmethod
    def zero_to_none(cls, v: int | None) -> int | None:
        """Treat 0 as None (models.dev uses 0 to mean unknown/unlimited)."""
        return None if v == 0 else v

    @field_validator("regions", mode="before")
    @classmethod
    def normalize_regions(cls, value: list[dict[str, str] | str] | None) -> list[dict[str, str] | str]:
        """Normalize string-only provider regions into the structured representation."""
        return [{"code": region, "name": region} if isinstance(region, str) else region for region in (value or [])]

    # ── Capability properties ──────────────────────────────────────────────

    @property
    def has_vision(self) -> bool:
        """True if model accepts image inputs."""
        return "image" in self.modalities_input

    @property
    def has_thinking(self) -> bool:
        """True if model natively supports chain-of-thought reasoning."""
        return self.reasoning

    @property
    def has_structured_outputs(self) -> bool:
        """True if model supports structured JSON output."""
        return self.structured_output

    @property
    def has_pdf(self) -> bool:
        """True if model accepts PDF inputs."""
        return "pdf" in self.modalities_input

    @property
    def has_audio(self) -> bool:
        """True if model accepts audio inputs."""
        return "audio" in self.modalities_input

    @property
    def has_video(self) -> bool:
        """True if model accepts video inputs."""
        return "video" in self.modalities_input

    @property
    def capabilities(self) -> list[str]:
        """Ordered capability list derived from models.dev flags and modality data."""
        caps: list[str] = []
        if self.has_vision:
            caps.append("vision")
        if self.has_thinking:
            caps.append("thinking")
        if self.has_structured_outputs:
            caps.append("structured_outputs")
        if self.has_pdf:
            caps.append("pdf")
        if self.has_audio:
            caps.append("audio")
        if self.has_video:
            caps.append("video")
        return caps


class ModelsDb:
    """Local cache and access layer for the models.dev database.

    Load with ``load()`` (reads from disk, fetches if absent) or directly
    call ``fetch()`` to download a fresh copy.
    """

    def __init__(self) -> None:
        self._index: dict[str, ModelEntry] = {}  # "provider_id/model_id" → ModelEntry
        self._providers: dict[str, dict[str, ModelEntry]] = {}  # provider_id → {model_id → entry}
        self._loaded = False
        self._cache_path: Path | None = None
        self._edenai_cache_paths: dict[str, Path] = {}

    # ── Loading / fetching ────────────────────────────────────────────────

    def load(self, cache_path: Path | None = None) -> "ModelsDb":
        """Load from local cache file, fetching automatically if absent."""
        if cache_path is None:
            cache_path = _default_cache_path()
        self._cache_path = cache_path
        if not cache_path.exists():
            logger.info(f"models.dev cache not found at {cache_path} — fetching now …")
            self.fetch(cache_path)
        else:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            self._build_index(raw)
        for provider_id, (_url, filename) in EDENAI_CATALOGUES.items():
            self._load_edenai_models(provider_id, cache_path.with_name(filename))
        self._loaded = True
        return self

    def fetch(self, cache_path: Path | None = None) -> "ModelsDb":
        """Download the latest models.dev database and save to the cache file."""
        if cache_path is None:
            cache_path = self._cache_path or _default_cache_path()
        logger.info(f"Fetching models.dev from {MODELS_DEV_URL} …")
        response = httpx.get(MODELS_DEV_URL, timeout=30)
        response.raise_for_status()
        raw: dict = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        total = sum(len(p.get("models", {})) for p in raw.values() if isinstance(p, dict))
        logger.info(f"Saved {len(raw)} providers / {total} models to {cache_path}")
        self._build_index(raw)
        self._loaded = True
        self._cache_path = cache_path
        return self

    def _load_edenai_models(self, provider_id: str, cache_path: Path) -> None:
        """Merge cached EdenAI models, fetching them only when a key is available."""
        self._edenai_cache_paths[provider_id] = cache_path
        if cache_path.exists():
            self._merge_edenai_models(provider_id, json.loads(cache_path.read_text(encoding="utf-8")))
        elif os.environ.get(EDENAI_API_KEY_ENV_VAR):
            self.fetch_edenai(cache_path, provider_id)
        else:
            logger.debug(f"EDENAI_API_KEY is not set; skipping {provider_id} model catalogue fetch.")

    def fetch_edenai(self, cache_path: Path | None = None, provider_id: str = "edenai") -> bool:
        """Fetch and cache an EdenAI model catalogue when an API key is configured."""
        api_key = os.environ.get(EDENAI_API_KEY_ENV_VAR)
        if not api_key:
            logger.debug(f"EDENAI_API_KEY is not set; skipping {provider_id} model catalogue fetch.")
            return False

        try:
            models_url, cache_filename = EDENAI_CATALOGUES[provider_id]
        except KeyError as error:
            raise ValueError(f"Unknown EdenAI catalogue provider: {provider_id}") from error
        if cache_path is None:
            cache_path = self._edenai_cache_paths.get(provider_id) or _default_cache_path().with_name(cache_filename)
        logger.info(f"Fetching {provider_id} models from {models_url} …")
        response = httpx.get(models_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        response.raise_for_status()
        raw: dict = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        self._edenai_cache_paths[provider_id] = cache_path
        self._merge_edenai_models(provider_id, raw)
        logger.info(f"Saved {len(raw.get('data', []))} {provider_id} models to {cache_path}")
        return True

    # ── Index construction ────────────────────────────────────────────────

    def _build_index(self, data: dict) -> None:
        self._index = {}
        self._providers = {}
        for provider_id, provider_data in data.items():
            if not isinstance(provider_data, dict):
                continue
            models = provider_data.get("models")
            if not isinstance(models, dict):
                continue
            self._providers[provider_id] = {}
            for model_id, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue
                entry = self._parse_entry(model_id, provider_id, model_data)
                self._index[f"{provider_id}/{model_id}"] = entry
                self._providers[provider_id][model_id] = entry

    def _merge_edenai_models(self, provider_id: str, data: dict) -> None:
        """Merge the EdenAI `/v3/models` response into the normalized index."""
        models = data.get("data")
        if not isinstance(models, list):
            logger.warning("Ignoring EdenAI model cache with an invalid 'data' field.")
            return

        edenai_models = self._providers.setdefault(provider_id, {})
        for model_data in models:
            if not isinstance(model_data, dict) or not isinstance(model_id := model_data.get("id"), str):
                continue
            entry = self._parse_edenai_entry(model_id, provider_id, model_data)
            self._index[f"{provider_id}/{model_id}"] = entry
            edenai_models[model_id] = entry

    def _parse_entry(self, model_id: str, provider_id: str, data: dict) -> ModelEntry:
        modalities = data.get("modalities") or {}
        limits = data.get("limit") or {}
        costs = data.get("cost") or {}
        return ModelEntry(
            id=model_id,
            name=data.get("name", model_id),
            provider_id=provider_id,
            family=data.get("family") or None,
            attachment=bool(data.get("attachment", False)),
            reasoning=bool(data.get("reasoning", False)),
            tool_call=bool(data.get("tool_call", False)),
            structured_output=bool(data.get("structured_output", False)),
            temperature=bool(data.get("temperature", True)),
            modalities_input=list(modalities.get("input") or []),
            modalities_output=list(modalities.get("output") or []),
            context=limits.get("context"),
            output=limits.get("output"),
            cost_input=float(costs["input"]) if costs.get("input") is not None else None,
            cost_output=float(costs["output"]) if costs.get("output") is not None else None,
            release_date=data.get("release_date") or None,
            last_updated=data.get("last_updated") or None,
            open_weights=bool(data.get("open_weights", False)),
            regions=list(data.get("regions") or []),
        )

    def _parse_edenai_entry(self, model_id: str, provider_id: str, data: dict) -> ModelEntry:
        capabilities = data.get("capabilities") or {}
        input_modalities = list(capabilities.get("input_modalities") or [])
        output_modalities = list(capabilities.get("output_modalities") or [])
        pricing = data.get("pricing") or {}
        return ModelEntry(
            id=model_id,
            name=data.get("model_name") or data.get("name") or model_id,
            provider_id=provider_id,
            family=data.get("owned_by") or None,
            reasoning=bool(capabilities.get("supports_reasoning", capabilities.get("reasoning", False))),
            tool_call=bool(capabilities.get("supports_function_calling", capabilities.get("tool_calling", False))),
            structured_output=bool(capabilities.get("supports_response_schema", False)),
            modalities_input=input_modalities + (["pdf"] if capabilities.get("pdf", False) else []),
            modalities_output=output_modalities,
            context=data.get("context_length"),
            cost_input=(float(pricing["input_cost_per_token"]) * 1_000_000)
            if pricing.get("input_cost_per_token") is not None
            else None,
            cost_output=(float(pricing["output_cost_per_token"]) * 1_000_000)
            if pricing.get("output_cost_per_token") is not None
            else None,
            regions=list(data.get("regions") or []),
        )

    # ── Lookup API ────────────────────────────────────────────────────────

    def lookup(self, provider_id: str, model_id: str) -> ModelEntry | None:
        """Look up a model by provider and model ID.

        Version suffixes like ``:exacto`` or ``:free`` are stripped before lookup.
        For gateway providers the model_id may be ``vendor/model-name``; in that case
        a fallback lookup in the vendor's own provider section is also attempted.

        Example:
            ```python
            db.lookup("openai", "gpt-4o-mini")
            db.lookup("openrouter", "openai/gpt-4.1-mini")
            db.lookup("litellm", "google/gemini-3-flash-preview")
            ```
        """
        model_key = model_id.split(":")[0]  # strip :exacto, :free etc.

        # Direct lookup (also covers openrouter entries like "openai/gpt-4o")
        entry = self._index.get(f"{provider_id}/{model_key}")
        if entry:
            return entry

        # For vendor-prefixed ids (openrouter / litellm style), fall back to the
        # direct vendor provider section in models.dev
        if "/" in model_key:
            vendor, _, model_suffix = model_key.partition("/")
            entry = self._index.get(f"{vendor}/{model_suffix}")
            if entry:
                return entry

        return None

    def provider_models(self, provider_id: str) -> dict[str, ModelEntry]:
        """Return {model_id: ModelEntry} for all models belonging to *provider_id*."""
        return dict(self._providers.get(provider_id, {}))

    def all_entries(self) -> dict[str, ModelEntry]:
        """Return flat dict of all entries keyed by ``provider_id/model_id``."""
        return dict(self._index)

    def stats(self) -> dict[str, int]:
        """Return model count per provider."""
        return {pid: len(models) for pid, models in self._providers.items()}

    @property
    def cache_path(self) -> Path | None:
        """Path to the local cache file used by this instance."""
        return self._cache_path


# ── Singleton ─────────────────────────────────────────────────────────────────


@once
def get_models_db() -> ModelsDb:
    """Return the singleton ``ModelsDb``, loading from the default cache path.

    The database is loaded once and cached in memory.  Call ``get_models_db.invalidate()``
    followed by ``get_models_db()`` to reload after a ``fetch()``.
    """
    return ModelsDb().load()
