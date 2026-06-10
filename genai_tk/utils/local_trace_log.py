"""Local JSONL trace logger for LLM calls.

Records prompts, responses, token usage, latency, and estimated cost to a
newline-delimited JSON file. Runs in parallel with any remote tracing backends.

Each log entry is a JSON object::

    {
        "ts": "2026-06-10T12:34:56.789+00:00",
        "session_id": "a1b2c3d4-...",
        "model": "gpt-4o-mini",
        "framework": "langchain",
        "prompt": "Tell me a joke …",
        "response": "Why did the …",
        "tokens_in": 42,
        "tokens_out": 18,
        "cost_usd": 0.00000840,
        "latency_ms": 412.3,
        "error": null,
    }

Usage as a LangChain callback is automatic when the ``local`` backend is active
in ``monitoring.backends``. For other frameworks use the helpers::

    from genai_tk.utils.local_trace_log import log_llm_call, baml_log_usage
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from loguru import logger
from pydantic import BaseModel

# Session UUID is fixed for the life of the process
SESSION_ID: str = str(uuid.uuid4())


class TraceEntry(BaseModel):
    """Single LLM call trace record."""

    ts: str
    session_id: str
    model: str
    framework: str = "langchain"
    prompt: str | None = None
    response: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    error: str | None = None


class LocalTraceLog(BaseCallbackHandler):
    """LangChain callback handler that appends LLM call traces to a JSONL file.

    Obtain the singleton via ``get_instance()``; do not instantiate directly.
    """

    _instance: LocalTraceLog | None = None

    def __init__(self, config: Any) -> None:  # config: LocalLogConfig
        super().__init__()
        self._config = config
        self._start_times: dict[str, float] = {}
        self._prompts: dict[str, str] = {}
        self._path = Path(config.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, config: Any | None = None) -> LocalTraceLog:
        """Return (or create) the singleton handler.

        Args:
            config: ``LocalLogConfig`` instance. When omitted the active config is used.
        """
        if cls._instance is None:
            if config is None:
                from genai_tk.utils.tracing import monitoring_config

                config = monitoring_config().local_log
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (used in tests)."""
        cls._instance = None

    # ── LangChain callback hooks ───────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.monotonic()
        if self._config.include_prompts and prompts:
            self._prompts[run_id] = _truncate("\n".join(prompts), self._config.max_prompt_chars)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.monotonic()
        if self._config.include_prompts and messages:
            flat = " | ".join(f"{m.type}: {getattr(m, 'content', '')}" for batch in messages for m in batch)
            self._prompts[run_id] = _truncate(flat, self._config.max_prompt_chars)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        latency_ms = _pop_latency(self._start_times, run_id)
        prompt_text = self._prompts.pop(run_id, None)

        llm_output = response.llm_output or {}
        model = llm_output.get("model_name", "") or ""
        token_usage = llm_output.get("token_usage", {}) or {}
        tokens_in: int | None = token_usage.get("prompt_tokens")
        tokens_out: int | None = token_usage.get("completion_tokens")

        response_text: str | None = None
        if self._config.include_prompts:
            gens = response.generations
            if gens and gens[0]:
                response_text = _truncate(
                    getattr(gens[0][0], "text", ""),
                    self._config.max_prompt_chars,
                )

        self._write(
            TraceEntry(
                ts=_now(),
                session_id=SESSION_ID,
                model=model,
                framework="langchain",
                prompt=prompt_text,
                response=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=_estimate_cost(model, tokens_in, tokens_out),
                latency_ms=latency_ms,
            )
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        latency_ms = _pop_latency(self._start_times, run_id)
        self._prompts.pop(run_id, None)
        self._write(
            TraceEntry(
                ts=_now(),
                session_id=SESSION_ID,
                model="",
                framework="langchain",
                error=str(error),
                latency_ms=latency_ms,
            )
        )

    # ── Internal write ─────────────────────────────────────────────────────────

    def _write(self, entry: TraceEntry) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning(f"local_trace_log write failed: {exc}")


# ── Standalone helpers (LiteLLM, BAML, custom) ────────────────────────────────


def log_llm_call(
    model: str,
    framework: str = "litellm",
    *,
    prompt: str | None = None,
    response: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Write a single LLM call record to the local JSONL log.

    Suitable for use as a LiteLLM success/failure callback or from BAML collectors.
    Does nothing if the local log singleton is not yet initialised.

    Args:
        model: Model identifier string.
        framework: Source framework name (``litellm``, ``baml``, etc.).
        prompt: Optional prompt text (truncated per config).
        response: Optional response text (truncated per config).
        tokens_in: Input token count.
        tokens_out: Output token count.
        latency_ms: Wall-clock latency in milliseconds.
        error: Error message if the call failed.
    """
    handler = LocalTraceLog._instance
    if handler is None:
        return
    cfg = handler._config
    max_chars = cfg.max_prompt_chars if cfg else 2000
    include = cfg.include_prompts if cfg else True

    handler._write(
        TraceEntry(
            ts=_now(),
            session_id=SESSION_ID,
            model=model,
            framework=framework,
            prompt=_truncate(prompt, max_chars) if (include and prompt) else None,
            response=_truncate(response, max_chars) if (include and response) else None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_estimate_cost(model, tokens_in, tokens_out),
            latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
            error=error,
        )
    )


def baml_log_usage(collector: Any, model: str, prompt_snippet: str | None = None) -> None:
    """Log token usage from a BAML collector to the local JSONL log.

    Args:
        collector: BAML ``BamlTracer`` or ``Collector`` instance.
        model: Model identifier string.
        prompt_snippet: Optional truncated prompt text to store.
    """
    if LocalTraceLog._instance is None:
        return
    try:
        usage = getattr(collector, "usage", None) or {}
        tokens_in: int | None = getattr(usage, "input_tokens", None) or (
            usage.get("input_tokens") if isinstance(usage, dict) else None
        )
        tokens_out: int | None = getattr(usage, "output_tokens", None) or (
            usage.get("output_tokens") if isinstance(usage, dict) else None
        )
        log_llm_call(
            model=model,
            framework="baml",
            prompt=prompt_snippet,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    except Exception as exc:
        logger.warning(f"baml_log_usage failed: {exc}")


def get_session_id() -> str:
    """Return the current process-scoped session UUID."""
    return SESSION_ID


# ── Private helpers ────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _truncate(text: str | None, max_chars: int) -> str | None:
    if not text:
        return None
    return (text[:max_chars] + "…") if len(text) > max_chars else text


def _pop_latency(start_times: dict[str, float], run_id: str) -> float | None:
    t = start_times.pop(run_id, None)
    if t is None:
        return None
    return round((time.monotonic() - t) * 1000, 1)


def _estimate_cost(model: str, tokens_in: int | None, tokens_out: int | None) -> float | None:
    """Best-effort cost estimate using the models.dev pricing database."""
    if not model or (tokens_in is None and tokens_out is None):
        return None
    try:
        from genai_tk.core.models_db import get_models_db

        db = get_models_db()
        entry = None
        # Match by model ID suffix (handles "provider/model-name" style strings)
        model_lower = model.lower()
        for e in db.all_entries().values():
            if model_lower.endswith(e.id.lower()) or e.id.lower() in model_lower:
                entry = e
                break
        if entry and (entry.cost_input is not None or entry.cost_output is not None):
            cost = 0.0
            if tokens_in and entry.cost_input:
                cost += tokens_in * entry.cost_input / 1_000_000
            if tokens_out and entry.cost_output:
                cost += tokens_out * entry.cost_output / 1_000_000
            return round(cost, 8) if cost > 0 else None
    except Exception:
        pass
    return None
