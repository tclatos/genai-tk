"""LangChain BaseChatModel adapter for genai-tk LLM identifiers.

When deepagents-cli's TUI ``/model`` switcher picks a model from the
``genai_tk`` provider, it calls::

    GenaiTkModelAdapter(model="<genai-tk-id>")

This class resolves that identifier through ``LlmFactory`` and delegates all
LangChain calls to the underlying model, giving deepagents-cli full access to
any LLM registered in the genai-tk configuration — without touching
deepagents-cli's own model-creation pipeline.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr


class GenaiTkModelAdapter(BaseChatModel):
    """Wraps any genai-tk LLM identifier as a LangChain ``BaseChatModel``.

    Used by deepagents-cli's ``/model`` TUI switcher via the
    ``class_path = "genai_tk.agents.deepagent.model_adapter:GenaiTkModelAdapter"``
    provider config entry.  The ``model`` field stores the genai-tk LLM tag or
    ID (e.g. ``"default"``, ``"fast_model"``, ``"gpt41mini@openai"``).
    """

    model: str
    """genai-tk LLM identifier (tag or explicit ID like ``gpt41mini@openai``)."""

    _delegate: BaseChatModel = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Resolve the genai-tk identifier to a concrete BaseChatModel."""
        from genai_tk.agents.deepagent.llm_bridge import _resolve_identifier

        self._delegate = _resolve_identifier(self.model)

    # ------------------------------------------------------------------
    # Required BaseChatModel interface
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "genai_tk_adapter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._delegate._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self._delegate._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from self._delegate._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        async for chunk in self._delegate._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    # ------------------------------------------------------------------
    # Pass-through bind/with_config so tool-calling wrappers propagate
    # ------------------------------------------------------------------

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._delegate.bind_tools(tools, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._delegate.with_structured_output(schema, **kwargs)

    # ------------------------------------------------------------------
    # Expose delegate's properties for introspection / logging
    # ------------------------------------------------------------------

    @property
    def _default_params(self) -> dict[str, Any]:
        return getattr(self._delegate, "_default_params", {"model": self.model})

    @property
    def model_name(self) -> str:  # type: ignore[override]
        name = getattr(self._delegate, "model_name", None) or getattr(self._delegate, "model", self.model)
        return str(name)

    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "genai_tk_adapter": True}
