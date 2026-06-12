"""Monitoring and tracing utilities.

Supports LangSmith, LangFuse, OpenTelemetry (standalone), and local JSONL logging.
Multiple backends can be active simultaneously.

Configuration (in YAML)::

    monitoring:
      backends: [langfuse, local]
      project: MyProject
      langfuse:
        host: ${oc.env:LANGFUSE_HOST,http://localhost:3000}
        public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,}
        secret_key: ${oc.env:LANGFUSE_SECRET_KEY,}
      otel:
        endpoint: ${oc.env:OTEL_EXPORTER_OTLP_ENDPOINT,http://localhost:4318}
      local_log:
        path: ${paths.data_root}/traces/llm_calls.jsonl
        include_prompts: true

Call ``setup_monitoring()`` once at application startup (the CLI does this
automatically). Use ``get_monitoring_callbacks()`` to obtain active LangChain
callbacks and ``monitoring_config()`` for typed config access.

Legacy: ``monitoring: {langsmith: true}`` still works unchanged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_tk.config_mgmt.config_mngr import global_config

# ── Sub-config models ──────────────────────────────────────────────────────────


class LangSmithBackendConfig(BaseModel):
    """LangSmith-specific settings."""

    endpoint: str = Field("https://api.smith.langchain.com")
    model_config = ConfigDict(extra="allow")


class LangFuseBackendConfig(BaseModel):
    """LangFuse host, OTEL endpoint, and API keys."""

    host: str = Field("http://localhost:3000")
    otel_host: str = Field("")
    public_key: str = Field("")
    secret_key: str = Field("")
    model_config = ConfigDict(extra="allow")

    @property
    def otel_endpoint(self) -> str:
        """OTEL endpoint derived from otel_host or host."""
        base = self.otel_host or self.host
        return f"{base.rstrip('/')}/api/public/otel"

    @property
    def is_configured(self) -> bool:
        """True if both public and secret keys are present."""
        return bool(self.public_key and self.secret_key)


class OtelBackendConfig(BaseModel):
    """Standalone OpenTelemetry collector settings."""

    endpoint: str = Field("http://localhost:4318")
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str = Field("genai-tk")
    model_config = ConfigDict(extra="allow")


class LocalLogConfig(BaseModel):
    """Local JSONL trace log settings."""

    enabled: bool = Field(True)
    path: str = Field("data/traces/llm_calls.jsonl")
    include_prompts: bool = Field(True)
    max_prompt_chars: int = Field(2000)
    model_config = ConfigDict(extra="allow")


class MonitoringConfig(BaseModel):
    """Typed view of the ``monitoring`` config section.

    Supports both the modern multi-backend form and the legacy ``langsmith: true`` flag.
    """

    backends: list[str] = Field(default_factory=list)
    project: str = Field("genai-tk")
    langsmith: LangSmithBackendConfig = Field(default_factory=LangSmithBackendConfig)
    langfuse: LangFuseBackendConfig = Field(default_factory=LangFuseBackendConfig)
    otel: OtelBackendConfig = Field(default_factory=OtelBackendConfig)
    local_log: LocalLogConfig = Field(default_factory=LocalLogConfig)
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy(cls, data: Any) -> Any:
        """Convert legacy ``langsmith: true`` to ``backends: [langsmith]``."""
        if not isinstance(data, dict):
            return data
        ls = data.get("langsmith")
        if ls is True:
            backends = list(data.get("backends") or [])
            if "langsmith" not in backends:
                backends.append("langsmith")
            data = {**data, "backends": backends, "langsmith": {}}
        elif ls is False:
            data = {**data, "langsmith": {}}
        return data

    def is_active(self, backend: str) -> bool:
        """Return True if the given backend is in the active list."""
        return backend in self.backends


# ── Runtime context ────────────────────────────────────────────────────────────


@dataclass
class MonitoringContext:
    """Runtime state created by ``setup_monitoring()``.

    ``langchain_callbacks`` holds explicitly registered LangChain callbacks
    (currently the local JSONL handler). Remote backends (LangSmith, LangFuse)
    work via env-var / OTEL auto-instrumentation and do not require explicit callbacks.
    """

    active_backends: list[str] = field(default_factory=list)
    langchain_callbacks: list = field(default_factory=list)

    def is_active(self, backend: str) -> bool:
        """Return True if the given backend is active."""
        return backend in self.active_backends


_monitoring_context: MonitoringContext | None = None


# ── Public API ─────────────────────────────────────────────────────────────────


def monitoring_config() -> MonitoringConfig:
    """Return typed monitoring settings from config."""
    try:
        return global_config().section("monitoring", MonitoringConfig)
    except Exception:
        return MonitoringConfig()


def should_enable_tracing() -> bool:
    """Return True if LangSmith tracing is active (backward-compatible check).

    Returns:
        True if the langsmith backend is active and the API key is present.
    """
    cfg = monitoring_config()
    if not cfg.is_active("langsmith"):
        return False
    try:
        from langsmith.utils import tracing_is_enabled

        return bool(tracing_is_enabled())
    except ImportError:
        return bool(os.environ.get("LANGSMITH_API_KEY"))


def get_monitoring_callbacks() -> list:
    """Return active LangChain callbacks (e.g. the local JSONL handler).

    Initialises monitoring on first call (lazy — avoids importing heavy packages
    at CLI startup when no LLM command is actually running).

    Remote backends use OTEL auto-instrumentation and do not appear here.
    Pass the returned list via ``config={"callbacks": ...}`` when invoking chains.
    """
    global _monitoring_context
    if _monitoring_context is None:
        try:
            setup_monitoring()
        except Exception as exc:
            logger.debug(f"Monitoring setup skipped: {exc}")
            return []
    return list(_monitoring_context.langchain_callbacks)


def setup_monitoring() -> MonitoringContext:
    """Initialise all active monitoring backends.

    Safe to call multiple times — subsequent calls return the cached context.
    Call once at CLI / application startup before any LLM interaction.

    Returns:
        ``MonitoringContext`` with active backend list and LangChain callbacks.
    """
    global _monitoring_context
    if _monitoring_context is not None:
        return _monitoring_context

    cfg = monitoring_config()
    callbacks: list = []

    if cfg.is_active("langsmith"):
        _setup_langsmith(cfg)

    if cfg.is_active("langfuse"):
        lf_cb = _setup_langfuse(cfg)
        if lf_cb is not None:
            callbacks.append(lf_cb)

    if cfg.is_active("otel"):
        _setup_otel(cfg)

    if cfg.is_active("local"):
        cb = _setup_local_log(cfg)
        if cb is not None:
            callbacks.append(cb)

    if cfg.backends:
        logger.debug(f"Monitoring active backends: {cfg.backends}")

    _monitoring_context = MonitoringContext(
        active_backends=list(cfg.backends),
        langchain_callbacks=callbacks,
    )
    return _monitoring_context


def reset_monitoring() -> None:
    """Reset the monitoring singleton (used in tests)."""
    global _monitoring_context
    _monitoring_context = None


# ── Backend setup helpers ──────────────────────────────────────────────────────


def _setup_langsmith(cfg: MonitoringConfig) -> None:
    """Activate LangSmith tracing via environment variables."""
    project = cfg.project or "genai-tk"
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    if cfg.langsmith.endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", cfg.langsmith.endpoint)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    logger.debug(f"LangSmith tracing enabled → project={os.environ.get('LANGSMITH_PROJECT')}")


def _setup_langfuse(cfg: MonitoringConfig) -> Any | None:
    """Activate LangFuse tracing; returns a LangChain CallbackHandler.

    When keys are not configured the function sets env vars and returns early
    without importing any heavy LangFuse / LangChain packages, keeping CLI
    startup fast.
    """
    lf = cfg.langfuse
    if not lf.is_configured:
        logger.debug(
            "LangFuse backend selected but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set — "
            "skipping LangFuse initialisation (set keys to enable tracing)."
        )
        return None

    # Set LangFuse SDK env vars (used by the langfuse Python SDK and litellm)
    if lf.public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", lf.public_key)
    if lf.secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", lf.secret_key)

    # Resolve effective host: config value → LANGFUSE_BASE_URL → LANGFUSE_HOST → config default
    # LANGFUSE_BASE_URL is the canonical env var for the LangFuse v3 Python SDK
    effective_host = (
        lf.host
        if lf.host and lf.host != "http://localhost:3000"
        else os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or lf.host
    )
    if effective_host:
        os.environ.setdefault("LANGFUSE_HOST", effective_host)
        os.environ.setdefault("LANGFUSE_BASEURL", effective_host)  # LangFuse v2 SDK
        os.environ.setdefault("LANGFUSE_BASE_URL", effective_host)  # LangFuse v3 SDK

    # Route OTEL exporter to LangFuse's OTEL endpoint
    otel_endpoint = lf.otel_endpoint if lf.otel_host else f"{(effective_host or lf.host).rstrip('/')}/api/public/otel"
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", otel_endpoint)
    _set_otel_auth_header_langfuse(lf)
    os.environ.setdefault("OTEL_SERVICE_NAME", f"genai-tk-{cfg.project}")

    # Auto-instrument LangChain and SmolAgents via OpenInference
    _instrument_langchain_otel()
    _instrument_smolagents_otel()

    # Register LiteLLM → LangFuse OTEL callback
    _setup_litellm_langfuse()

    # ── LangFuse LangChain CallbackHandler (v4 SDK) ───────────────────────────
    # The OTEL BatchSpanProcessor is async and may not flush before a short-lived
    # CLI process exits.  The SDK's own CallbackHandler sends traces synchronously
    # and also calls flush() on its client, making it reliable for CLI use.
    handler: Any | None = None
    if lf.is_configured:
        try:
            from langfuse.langchain import CallbackHandler

            # Do NOT pass public_key — LangFuse v4 uses a singleton registry.
            # Passing public_key calls get_client(public_key=...) which looks up an
            # existing registered instance; if none exists it returns a DISABLED
            # fake client.  Calling CallbackHandler() with no args lets get_client()
            # create a fresh Langfuse() instance using the env vars set above.
            handler = CallbackHandler()
            # Flush pending spans when the process exits (important for short-lived CLI)
            import atexit

            lf_client = getattr(handler, "_langfuse_client", None)
            if lf_client is not None and hasattr(lf_client, "flush"):
                atexit.register(lf_client.flush)
            logger.debug(f"LangFuse CallbackHandler registered → {effective_host or lf.host}")
        except ImportError:
            logger.debug("langfuse.langchain not available — falling back to OTEL-only")
        except Exception as exc:
            logger.warning(f"LangFuse CallbackHandler setup failed: {exc}")

    logger.debug(f"LangFuse tracing enabled → {otel_endpoint}")
    return handler


def _set_otel_auth_header_langfuse(lf: LangFuseBackendConfig) -> None:
    """Build and set the OTEL Basic-auth header for LangFuse."""
    if not (lf.public_key and lf.secret_key):
        return
    import base64

    token = base64.b64encode(f"{lf.public_key}:{lf.secret_key}".encode()).decode()
    auth_header = f"Authorization=Basic {token}"
    existing = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    if "Authorization" not in existing:
        combined = f"{existing},{auth_header}" if existing else auth_header
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = combined


def _setup_otel(cfg: MonitoringConfig) -> None:
    """Activate standalone OpenTelemetry tracing."""
    otel = cfg.otel
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", otel.endpoint)
    os.environ.setdefault("OTEL_SERVICE_NAME", otel.service_name or f"genai-tk-{cfg.project}")
    if otel.headers:
        header_str = ",".join(f"{k}={v}" for k, v in otel.headers.items())
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", header_str)

    _instrument_langchain_otel()
    _instrument_smolagents_otel()
    logger.debug(f"OTEL tracing enabled → {os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')}")


def _instrument_langchain_otel() -> None:
    """Instrument LangChain via OpenInference OTEL instrumentor."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
        logger.debug("LangChain OTEL auto-instrumentation active")
    except ImportError:
        logger.debug("openinference-instrumentation-langchain not installed — skipping")
    except Exception as exc:
        logger.warning(f"LangChain OTEL instrumentation failed: {exc}")


def _instrument_smolagents_otel() -> None:
    """Instrument SmolAgents via OpenInference OTEL instrumentor."""
    try:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        SmolagentsInstrumentor().instrument()
        logger.debug("SmolAgents OTEL auto-instrumentation active")
    except ImportError:
        logger.debug("openinference-instrumentation-smolagents not installed — skipping")
    except Exception as exc:
        logger.warning(f"SmolAgents OTEL instrumentation failed: {exc}")


def _setup_litellm_langfuse() -> None:
    """Register LiteLLM → LangFuse OTEL callback — only if litellm is already loaded.

    Importing litellm on a cold start takes ~2.5s. We skip the registration
    entirely when litellm hasn't been imported yet; it will be registered the
    first time the caller actually imports litellm (e.g. when running an LLM).
    """
    import sys

    if "litellm" not in sys.modules:
        logger.debug("LiteLLM not yet loaded — skipping langfuse_otel callback registration")
        return
    try:
        import litellm

        if "langfuse_otel" not in (litellm.callbacks or []):
            litellm.callbacks = list(litellm.callbacks or []) + ["langfuse_otel"]
        logger.debug("LiteLLM langfuse_otel callback registered")
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"LiteLLM LangFuse callback setup failed: {exc}")


def _setup_local_log(cfg: MonitoringConfig) -> Any | None:
    """Initialise the local JSONL logger; returns the LangChain callback handler."""
    if not cfg.local_log.enabled:
        return None
    try:
        from genai_tk.utils.local_trace_log import LocalTraceLog

        handler = LocalTraceLog.get_instance(cfg.local_log)
        logger.debug(f"Local JSONL trace log active → {cfg.local_log.path}")
        return handler
    except Exception as exc:
        logger.warning(f"Local trace log setup failed: {exc}")
        return None


# ── Backward-compatibility shims ───────────────────────────────────────────────


@contextmanager
def tracing_context() -> Generator[None, None, None]:
    """Context manager kept for backward compatibility.

    Tracing is controlled by env vars / OTEL instrumentation set up in
    ``setup_monitoring()``. This context manager is a no-op.

    Yields:
        None
    """
    yield None
