"""Middleware that marks the static prompt prefix as cacheable.

Benchmark and chat agents resend the same large system prompt (including
injected skill text) on every model call of every question. Providers that
support explicit prompt caching (Anthropic — also via OpenRouter's
``cache_control`` passthrough) bill cached prefix tokens at a steep discount
and return faster time-to-first-token.

This middleware tags the leading ``SystemMessage`` with an Anthropic-style
``cache_control`` breakpoint (``additional_kwargs={"cache_control": {...}}``),
which ``langchain-anthropic`` serializes into the API request. Providers whose
LangChain integration ignores ``additional_kwargs`` (OpenAI-compatible
endpoints) simply never see the field — the middleware is a safe no-op there.
Place the middleware **last** in the profile's ``middlewares`` list so it sees
the fully-assembled request after skill/prompt-injection middlewares ran.

Example YAML config::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.prompt_cache.PromptCacheMiddleware
        cache_control: {"type": "ephemeral"}
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from loguru import logger

DEFAULT_CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral"}


class PromptCacheMiddleware(AgentMiddleware):
    """Tag the leading system message with a provider prompt-cache breakpoint.

    Args:
        cache_control: Cache-control payload attached to the system message.
            Defaults to the Anthropic ephemeral breakpoint.
        tag_all_system: Tag every ``SystemMessage`` instead of only the leading
            one. Off by default — later system messages are usually dynamic
            (nudges, directives) and must stay outside the cached prefix.
    """

    def __init__(
        self,
        cache_control: dict[str, Any] | None = None,
        *,
        tag_all_system: bool = False,
    ) -> None:
        self._cache_control = cache_control or dict(DEFAULT_CACHE_CONTROL)
        self._tag_all_system = tag_all_system
        self._logged = False

    def _tag(self, request: ModelRequest) -> ModelRequest:
        messages = request.messages
        if not messages:
            return request

        tagged = False
        new_messages: list[Any] = []
        for i, msg in enumerate(messages):
            already_tagged = isinstance(msg.additional_kwargs.get("cache_control"), dict)
            is_target = isinstance(msg, SystemMessage) and (self._tag_all_system or not tagged)
            if is_target and not already_tagged:
                new_messages.append(
                    msg.model_copy(update={"additional_kwargs": {**msg.additional_kwargs, "cache_control": self._cache_control}})
                )
                tagged = True
            else:
                new_messages.append(msg)
            if not self._tag_all_system and tagged:
                new_messages.extend(messages[i + 1 :])
                break

        if not tagged:
            return request
        if not self._logged:
            logger.debug("[PromptCache] Tagged system prompt with cache_control {}", self._cache_control)
            self._logged = True
        return request.override(messages=new_messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant — tag the cacheable prefix, then call the model."""
        return handler(self._tag(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant — same logic as sync."""
        return await handler(self._tag(request))
