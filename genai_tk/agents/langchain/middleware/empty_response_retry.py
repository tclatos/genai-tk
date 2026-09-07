"""Middleware to detect and retry empty LLM responses.

Some models (especially reasoning/thinking models) may consume reasoning tokens
but produce no visible content and no tool calls. The LangChain agent loop
interprets this as "done" and exits — yielding an empty response to the user.

This middleware detects that pattern inside ``awrap_model_call`` and retries
the request, optionally swapping to a fallback model on the final attempt.

Example YAML config::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.empty_response_retry.EmptyResponseRetryMiddleware
        max_retries: 2
        fallback_llm: claude-sonnet@openrouter
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from genai_tk.core.messages import extract_ai_message_parts

DEFAULT_RECOVERY_PROMPT: str = (
    "Your previous response was completely empty (no text and no tool calls). "
    "Please continue your analysis and provide your response, emit the next required tool call, "
    "or deliver your final answer."
)


def _unwrap_ai_message(response: Any) -> AIMessage | None:
    """Extract the AIMessage from a ModelResponse / ExtendedModelResponse."""
    inner = response
    if hasattr(inner, "model_response"):
        inner = inner.model_response
    if hasattr(inner, "result"):
        msgs = inner.result
        if msgs:
            msg = msgs[0]
            if isinstance(msg, AIMessage):
                return msg
    if isinstance(inner, AIMessage):
        return inner
    return None


def _is_empty(msg: AIMessage) -> bool:
    """True when the AIMessage carries neither visible text content nor tool calls."""
    parts = extract_ai_message_parts(msg)
    return len(parts.text.strip()) == 0 and len(parts.tool_calls) == 0


class EmptyResponseRetryMiddleware(AgentMiddleware):
    """Retry LLM calls that produce empty responses (no content, no tool calls).

    On each retry the original model is called again. On the **last** attempt
    the request is re-issued with ``fallback_model`` (if provided) so that a
    more capable or differently-configured model can recover the conversation.
    If ``recovery_prompt`` is set, an active recovery prompt is injected into the
    messages on each retry attempt to guide the model back to generating output.

    Args:
        max_retries: Number of additional attempts after the first empty response.
        fallback_model: Pre-built ``BaseChatModel`` to use on the last retry.
        fallback_llm: LLM identifier resolved via ``LlmFactory`` (e.g. ``claude-sonnet@openrouter``).
            Used when ``fallback_model`` is not provided.  Takes effect only on the last retry.
        recovery_prompt: Optional recovery instruction injected as a HumanMessage on retry attempts.
            Defaults to ``DEFAULT_RECOVERY_PROMPT``. Set to ``None`` or empty string to disable.
    """

    def __init__(
        self,
        max_retries: int = 1,
        fallback_model: BaseChatModel | None = None,
        fallback_llm: str | None = None,
        recovery_prompt: str | None = DEFAULT_RECOVERY_PROMPT,
    ) -> None:
        self._max_retries = max_retries
        self._fallback_model = fallback_model
        self._fallback_llm = fallback_llm
        self._recovery_prompt = recovery_prompt or None
        self._resolved_fallback: BaseChatModel | None = None

    def _get_fallback(self) -> BaseChatModel | None:
        if self._fallback_model is not None:
            return self._fallback_model
        if self._fallback_llm and self._resolved_fallback is None:
            from genai_tk.core.factories.llm_factory import get_llm

            self._resolved_fallback = get_llm(llm=self._fallback_llm)
        return self._resolved_fallback

    def _make_retry_request(self, request: ModelRequest, attempt: int) -> ModelRequest:
        if not hasattr(request, "override"):
            return request

        is_last = attempt == self._max_retries
        fallback = self._get_fallback() if is_last else None
        override_kwargs: dict[str, Any] = {}

        if fallback is not None:
            model_name = getattr(fallback, "model_name", None) or getattr(fallback, "model", "unknown")
            logger.warning(
                f"[EmptyResponseRetry] Empty LLM response (attempt {attempt}/{self._max_retries}) — "
                f"retrying with fallback model: {model_name}"
            )
            override_kwargs["model"] = fallback
        else:
            logger.warning(
                f"[EmptyResponseRetry] Empty LLM response (attempt {attempt}/{self._max_retries}) — "
                "retrying with same model"
            )

        if self._recovery_prompt and hasattr(request, "messages") and request.messages is not None:
            override_kwargs["messages"] = [*request.messages, HumanMessage(content=self._recovery_prompt)]
            logger.info(
                f"[EmptyResponseRetry] Injected active recovery prompt on attempt {attempt}/{self._max_retries}"
            )

        if override_kwargs:
            return request.override(**override_kwargs)
        return request

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Call handler; on empty response retry up to ``max_retries`` times."""
        response = await handler(request)
        msg = _unwrap_ai_message(response)
        if msg is None or not _is_empty(msg):
            return response

        for attempt in range(1, self._max_retries + 1):
            retry_request = self._make_retry_request(request, attempt)
            response = await handler(retry_request)
            msg = _unwrap_ai_message(response)
            if msg is None or not _is_empty(msg):
                return response

        logger.error(
            f"[EmptyResponseRetry] LLM returned empty response after "
            f"{self._max_retries + 1} attempt(s). Returning last empty response."
        )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Sync variant — same logic as async."""
        response = handler(request)
        msg = _unwrap_ai_message(response)
        if msg is None or not _is_empty(msg):
            return response

        for attempt in range(1, self._max_retries + 1):
            retry_request = self._make_retry_request(request, attempt)
            response = handler(retry_request)
            msg = _unwrap_ai_message(response)
            if msg is None or not _is_empty(msg):
                return response

        logger.error(
            f"[EmptyResponseRetry] LLM returned empty response after "
            f"{self._max_retries + 1} attempt(s). Returning last empty response."
        )
        return response
