"""Local cache and access layer for the models.dev database.

Downloads and caches the models.dev/api.json database locally and provides fast
lookup of model metadata — capabilities, context limits, pricing — without relying
on any third-party LangChain provider packages.

The database is stored in ``data/models_dev.json`` relative to the workspace root.
Use ``cli info llm-refresh --reload`` to update to the latest version from models.dev.

Example:
    ```python
    db = get_models_db()
    entry = db.lookup("openai", "gpt-4o-mini")
    if entry:
        print(entry.capabilities)
    ```
"""

import json
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel, field_validator

MODELS_DEV_URL = "https://models.dev/api.json"
_DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "models_dev.json"


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

    model_config = {"frozen": True}

    @field_validator("context", "output", mode="before")
    @classmethod
    def zero_to_none(cls, v: int | None) -> int | None:
        """Treat 0 as None (models.dev uses 0 to mean unknown/unlimited)."""
        return None if v == 0 else v

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

    # ── Loading / fetching ────────────────────────────────────────────────

    def load(self, cache_path: Path = _DEFAULT_CACHE_PATH) -> "ModelsDb":
        """Load from local cache file, fetching automatically if absent."""
        self._cache_path = cache_path
        if not cache_path.exists():
            logger.info(f"models.dev cache not found at {cache_path} — fetching now …")
            self.fetch(cache_path)
        else:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            self._build_index(raw)
        self._loaded = True
        return self

    def fetch(self, cache_path: Path | None = None) -> "ModelsDb":
        """Download the latest models.dev database and save to the cache file."""
        if cache_path is None:
            cache_path = self._cache_path or _DEFAULT_CACHE_PATH
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

_db_singleton: ModelsDb | None = None


def get_models_db() -> ModelsDb:
    """Return the singleton ``ModelsDb``, loading from the default cache path.

    The database is loaded once and cached in memory.  Call ``invalidate_models_db()``
    followed by ``get_models_db()`` to reload after a ``fetch()``.
    """
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = ModelsDb().load()
    return _db_singleton


def invalidate_models_db() -> None:
    """Discard the singleton so the next call to ``get_models_db()`` reloads from disk."""
    global _db_singleton
    _db_singleton = None
