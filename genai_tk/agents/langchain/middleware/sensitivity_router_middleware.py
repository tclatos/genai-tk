"""Sensitivity-aware LLM routing middleware for LangChain agents.

Routes model calls to a designated "safe" LLM when:
- The message content is assessed as sensitive by the configured :class:`SensitivityScorer`, **or**
- A tool has returned :class:`~langchain_core.documents.Document` objects whose
  ``source`` metadata path matches one of the configured ``sensitive_source_patterns``
  (glob-style).

The RAG-source detection works by intercepting tool call results and inspecting
the ``artifact`` field of ``ToolMessage`` objects for lists of LangChain
``Document`` objects.  Once a sensitive source is encountered, all subsequent
model calls for that ``thread_id`` are routed to the safe LLM.

Example YAML config::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.sensitivity_router_middleware:SensitivityRouterMiddleware
        safe_llm: ollama_local
        sensitive_source_patterns:
          - "**/hr/**"
          - "**/confidential/**"

Example programmatic usage::

    ```python
    from genai_tk.agents.langchain.middleware.sensitivity_router_middleware import (
        SensitivityRouterConfig,
        SensitivityRouterMiddleware,
    )

    middleware = SensitivityRouterMiddleware(
        SensitivityRouterConfig(safe_llm="ollama_local", sensitive_source_patterns=["**/hr/**"])
    )
    ```
"""

from __future__ import annotations

import fnmatch
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.agents.langchain.middleware.sensitivity_scorer import (
    DefaultSensitivityScorer,
    SensitivityScorer,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


_DEFAULT_THREAD = "__default__"


class SensitivityRouterConfig(BaseModel):
    """Configuration for :class:`SensitivityRouterMiddleware`.

    Can be passed directly to the constructor or expanded as keyword arguments
    when instantiated via the YAML middleware system.
    """

    scorer_class: str = Field(
        default="genai_tk.agents.langchain.middleware.sensitivity_scorer:DefaultSensitivityScorer",
        description=(
            "Qualified path to a class implementing the SensitivityScorer protocol "
            "(module.path:ClassName).  The class is instantiated with ``scorer_kwargs``."
        ),
    )
    scorer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the scorer constructor",
    )
    safe_llm: str = Field(
        description="LLM identifier or tag (e.g. 'ollama_local', 'safe_model') resolved via LlmFactory"
    )
    sensitive_source_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns for RAG document source paths that unconditionally trigger "
            "routing to the safe LLM (e.g. '**/hr/**', '**/confidential/**')"
        ),
    )


class SensitivityRouterMiddleware(AgentMiddleware):
    """Routes sensitive model calls to a safe LLM.

    Detection is two-pronged:
    1. **Content-based**: Each outgoing model call is assessed by the configured
       :class:`~genai_tk.agents.langchain.middleware.sensitivity_scorer.SensitivityScorer`.
       If the score exceeds the scorer's threshold, the call is re-routed.
    2. **Source-based**: Tool results are inspected for LangChain ``Document``
       objects (in ``ToolMessage.artifact``).  When a document's ``source``
       metadata matches one of ``sensitive_source_patterns``, all subsequent
       model calls for that thread are routed to the safe LLM regardless of content.

    Args:
        config: Full router config.  Alternatively, pass keyword arguments
            matching :class:`SensitivityRouterConfig` fields directly (YAML style).
    """

    def __init__(self, config: SensitivityRouterConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = SensitivityRouterConfig(**kwargs)
        self._config = config
        self._scorer: SensitivityScorer = self._build_scorer()
        self._safe_model: BaseChatModel | None = None  # lazy-resolved
        # sensitive_sources[thread_id] = set of matched source paths
        self._sensitive_sources: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Wrap-based hooks
    # ------------------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Assess sensitivity and swap to safe model if needed."""
        thread_id = self._thread_id(request)
        routed, reason = self._should_route(request, thread_id)
        if routed:
            safe_model = self._get_safe_model()
            logger.info(f"[SensitivityRouter] Routing to safe LLM — reason: {reason} (thread={thread_id!r})")
            request = request.override(model=safe_model)
        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant of :meth:`awrap_model_call`."""
        thread_id = self._thread_id(request)
        routed, reason = self._should_route(request, thread_id)
        if routed:
            safe_model = self._get_safe_model()
            logger.info(f"[SensitivityRouter] Routing to safe LLM — reason: {reason} (thread={thread_id!r})")
            request = request.override(model=safe_model)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Execute tool call and inspect result for sensitive Document sources."""
        result = await handler(request)
        self._inspect_tool_result(result, request)
        return result

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Sync variant of :meth:`awrap_tool_call`."""
        result = handler(request)
        self._inspect_tool_result(result, request)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_route(self, request: ModelRequest, thread_id: str) -> tuple[bool, str]:
        """Return (should_route, reason) for the given request."""
        # 1. Source-based routing (sticky per thread)
        if self._sensitive_sources.get(thread_id):
            return True, "sensitive RAG source previously encountered"

        # 2. Content-based routing
        if self._config.sensitive_source_patterns or True:  # always evaluate content
            text = self._extract_text(request)
            if text:
                assessment = self._scorer.assess(text)
                if assessment.is_sensitive:
                    return True, f"sensitive content detected (score={assessment.score:.2f}, level={assessment.level})"

        return False, ""

    def _inspect_tool_result(self, result: Any, request: Any) -> None:
        """Check if a tool result contains Documents from sensitive sources."""
        if not self._config.sensitive_source_patterns:
            return

        # Try to extract ToolMessage from result
        tool_message: ToolMessage | None = None
        if isinstance(result, ToolMessage):
            tool_message = result
        elif hasattr(result, "messages"):
            for msg in result.messages:
                if isinstance(msg, ToolMessage):
                    tool_message = msg
                    break

        if tool_message is None:
            return

        artifact = getattr(tool_message, "artifact", None)
        if not isinstance(artifact, list):
            return

        thread_id = self._thread_id_from_request(request)
        sources = self._sensitive_sources.setdefault(thread_id, set())

        for item in artifact:
            if not isinstance(item, Document):
                continue
            source = item.metadata.get("source", "")
            if source and self._matches_sensitive_pattern(source):
                logger.info(
                    f"[SensitivityRouter] Sensitive source detected: {source!r} "
                    f"(thread={thread_id!r}) — future calls routed to safe LLM"
                )
                sources.add(source)

    def _matches_sensitive_pattern(self, source: str) -> bool:
        """Return True if *source* matches any configured glob pattern."""
        normalized = source.replace("\\", "/")
        for pattern in self._config.sensitive_source_patterns:
            if fnmatch.fnmatch(normalized, pattern):
                return True
            # Also try matching just the path tail for patterns like '**/hr/**'
            if fnmatch.fnmatch("/" + normalized.lstrip("/"), pattern):
                return True
        return False

    def _extract_text(self, request: ModelRequest) -> str:
        """Extract human-readable text from the most recent human message."""
        messages = getattr(request, "messages", None) or []
        from langchain_core.messages import HumanMessage as HM

        for msg in reversed(messages):
            if isinstance(msg, HM):
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
                    return " ".join(parts)
        return ""

    def _thread_id(self, request: ModelRequest) -> str:
        try:
            from langgraph.config import get_config  # noqa: PLC0415

            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            pass
        try:
            cfg = getattr(request, "config", {}) or {}
            return cfg.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            return _DEFAULT_THREAD

    def _thread_id_from_request(self, request: Any) -> str:
        try:
            from langgraph.config import get_config  # noqa: PLC0415

            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            pass
        try:
            if hasattr(request, "config"):
                return request.config.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            pass
        return _DEFAULT_THREAD

    def _get_safe_model(self) -> BaseChatModel:
        """Lazily resolve the safe LLM (only when first needed)."""
        if self._safe_model is None:
            from genai_tk.core.llm_factory import get_llm

            self._safe_model = get_llm(llm=self._config.safe_llm)
        return self._safe_model

    def _build_scorer(self) -> SensitivityScorer:
        """Instantiate the scorer from the qualified class path."""
        if (
            self._config.scorer_class
            == "genai_tk.agents.langchain.middleware.sensitivity_scorer:DefaultSensitivityScorer"
            and not self._config.scorer_kwargs
        ):
            return DefaultSensitivityScorer()

        try:
            from genai_tk.utils.config_mngr import import_from_qualified

            cls = import_from_qualified(self._config.scorer_class)
            return cls(**self._config.scorer_kwargs)
        except Exception as exc:
            logger.warning(
                f"[SensitivityRouter] Failed to load scorer '{self._config.scorer_class}': {exc}. "
                "Falling back to DefaultSensitivityScorer."
            )
            return DefaultSensitivityScorer()
