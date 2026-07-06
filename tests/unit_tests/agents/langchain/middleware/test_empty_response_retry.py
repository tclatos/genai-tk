"""Unit tests for ``empty_response_retry.py``.

Exercises the empty-response detection helpers and the
``EmptyResponseRetryMiddleware`` retry loop (async + sync) using a fake handler
that returns canned responses in sequence.  The fallback model is a tiny stand-in
for an external LLM boundary; the fallback-LLM resolution path uses the real
fake LLM id (``parrot_local@fake``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from genai_tk.agents.langchain.middleware.empty_response_retry import (
    EmptyResponseRetryMiddleware,
    _is_empty,
    _unwrap_ai_message,
)

# --------------------------------------------------------------------------- #
# Fake collaborators
# --------------------------------------------------------------------------- #


class _FakeRequest:
    """Stand-in for a ModelRequest that records ``override`` calls."""

    def __init__(self) -> None:
        self.override_calls: list[dict[str, Any]] = []

    def override(self, **kw: Any) -> "_FakeRequest":
        self.override_calls.append(kw)
        return _FakeRequest()


class _FakeFallbackModel:
    """Stand-in for a fallback ``BaseChatModel``."""


def _model_response(msg: AIMessage) -> SimpleNamespace:
    """Wrap an AIMessage in a ModelResponse-like object (exercises the unwrap path)."""
    return SimpleNamespace(result=[msg])


def _empty() -> AIMessage:
    return AIMessage(content="")


def _nonempty(text: str = "ok") -> AIMessage:
    return AIMessage(content=text)


def _async_handler(responses: list[Any]) -> tuple[Any, list[Any]]:
    """Build an async handler that returns ``responses`` in order, recording calls."""
    calls: list[Any] = []

    async def _h(_req: Any) -> Any:
        idx = len(calls)
        calls.append(_req)
        return responses[idx]

    return _h, calls


def _sync_handler(responses: list[Any]) -> tuple[Any, list[Any]]:
    """Build a sync handler that returns ``responses`` in order, recording calls."""
    calls: list[Any] = []

    def _h(_req: Any) -> Any:
        idx = len(calls)
        calls.append(_req)
        return responses[idx]

    return _h, calls


# --------------------------------------------------------------------------- #
# _unwrap_ai_message
# --------------------------------------------------------------------------- #


def test_unwrap_direct_ai_message() -> None:
    msg = _nonempty("direct")
    assert _unwrap_ai_message(msg) is msg


def test_unwrap_model_response_with_ai_message() -> None:
    msg = _nonempty("wrapped")
    assert _unwrap_ai_message(_model_response(msg)) is msg


def test_unwrap_extended_model_response() -> None:
    msg = _nonempty("extended")
    wrapper = SimpleNamespace(model_response=_model_response(msg))
    assert _unwrap_ai_message(wrapper) is msg


def test_unwrap_empty_result_list_returns_none() -> None:
    assert _unwrap_ai_message(SimpleNamespace(result=[])) is None


def test_unwrap_non_ai_message_in_result_returns_none() -> None:
    assert _unwrap_ai_message(SimpleNamespace(result=[HumanMessage(content="hi")])) is None


def test_unwrap_unrelated_object_returns_none() -> None:
    assert _unwrap_ai_message(SimpleNamespace(foo="bar")) is None


# --------------------------------------------------------------------------- #
# _is_empty
# --------------------------------------------------------------------------- #


def test_is_empty_empty_string() -> None:
    assert _is_empty(_empty()) is True


def test_is_empty_whitespace_only() -> None:
    assert _is_empty(AIMessage(content="   \n\t  ")) is True


def test_is_empty_with_text_is_false() -> None:
    assert _is_empty(_nonempty("hi")) is False


def test_is_empty_with_tool_calls_is_false() -> None:
    msg = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1", "type": "tool_call"}])
    assert _is_empty(msg) is False


def test_is_empty_with_empty_text_block_list_is_true() -> None:
    assert _is_empty(AIMessage(content=[{"type": "text", "text": ""}])) is True


def test_is_empty_with_nonempty_text_block_list_is_false() -> None:
    assert _is_empty(AIMessage(content=[{"type": "text", "text": "hi"}])) is False


# --------------------------------------------------------------------------- #
# EmptyResponseRetryMiddleware — async retry behaviour
# --------------------------------------------------------------------------- #


async def test_awrap_non_empty_response_no_retry() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = _async_handler([_nonempty("ok")])

    response = await mw.awrap_model_call(_FakeRequest(), handler)

    assert response is not None
    assert len(calls) == 1


async def test_awrap_empty_then_non_empty_retries_until_success() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = _async_handler([_empty(), _nonempty("recovered")])

    response = await mw.awrap_model_call(_FakeRequest(), handler)

    assert _unwrap_ai_message(response) is not None
    assert _unwrap_ai_message(response).content == "recovered"  # type: ignore[union-attr]
    assert len(calls) == 2


async def test_awrap_empty_always_returns_last_empty() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=1)
    handler, calls = _async_handler([_empty(), _empty()])

    response = await mw.awrap_model_call(_FakeRequest(), handler)

    assert _is_empty(_unwrap_ai_message(response))  # type: ignore[arg-type]
    # 1 original call + 1 retry.
    assert len(calls) == 2


async def test_awrap_unwrap_none_returns_without_retry() -> None:
    """If the response cannot be unwrapped to an AIMessage, it's returned as-is."""
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = _async_handler([SimpleNamespace(foo="bar")])

    response = await mw.awrap_model_call(_FakeRequest(), handler)

    assert response is not None
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Fallback model / fallback LLM
# --------------------------------------------------------------------------- #


async def test_awrap_fallback_model_overrides_on_last_attempt() -> None:
    fallback = _FakeFallbackModel()
    mw = EmptyResponseRetryMiddleware(max_retries=1, fallback_model=fallback)
    request = _FakeRequest()
    handler, calls = _async_handler([_empty(), _empty()])

    await mw.awrap_model_call(request, handler)

    # The last (only) retry must swap in the fallback model via request.override.
    assert any(call.get("model") is fallback for call in request.override_calls)
    assert len(calls) == 2


async def test_awrap_fallback_model_only_on_last_attempt_with_multiple_retries() -> None:
    fallback = _FakeFallbackModel()
    mw = EmptyResponseRetryMiddleware(max_retries=2, fallback_model=fallback)
    request = _FakeRequest()
    handler, calls = _async_handler([_empty(), _empty(), _empty()])

    await mw.awrap_model_call(request, handler)

    # override is called only on the final attempt (attempt 2), not attempt 1.
    assert len(request.override_calls) == 1
    assert request.override_calls[0].get("model") is fallback
    assert len(calls) == 3


async def test_awrap_no_fallback_does_not_override() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    request = _FakeRequest()
    handler, _ = _async_handler([_empty(), _empty(), _nonempty("ok")])

    await mw.awrap_model_call(request, handler)

    assert request.override_calls == []


def test_get_fallback_with_fallback_model_returns_directly() -> None:
    fallback = _FakeFallbackModel()
    mw = EmptyResponseRetryMiddleware(fallback_model=fallback)

    assert mw._get_fallback() is fallback


def test_get_fallback_with_fallback_llm_resolves(fake_llm_id: str) -> None:
    mw = EmptyResponseRetryMiddleware(fallback_llm=fake_llm_id)

    resolved = mw._get_fallback()

    assert resolved is not None
    assert mw._resolved_fallback is resolved
    assert type(resolved).__name__ == "ParrotFakeChatModel"


def test_get_fallback_with_none_returns_none() -> None:
    mw = EmptyResponseRetryMiddleware()
    assert mw._get_fallback() is None


# --------------------------------------------------------------------------- #
# _make_retry_request
# --------------------------------------------------------------------------- #


def test_make_retry_request_non_last_returns_same_request() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    request = _FakeRequest()

    assert mw._make_retry_request(request, attempt=1) is request
    assert request.override_calls == []


def test_make_retry_request_last_without_fallback_returns_same_request() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=1)
    request = _FakeRequest()

    assert mw._make_retry_request(request, attempt=1) is request


def test_make_retry_request_last_with_fallback_overrides() -> None:
    fallback = _FakeFallbackModel()
    mw = EmptyResponseRetryMiddleware(max_retries=1, fallback_model=fallback)
    request = _FakeRequest()

    retry = mw._make_retry_request(request, attempt=1)

    assert retry is not request
    assert request.override_calls == [{"model": fallback}]


# --------------------------------------------------------------------------- #
# Sync wrap_model_call
# --------------------------------------------------------------------------- #


def test_wrap_sync_non_empty_response_no_retry() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = _sync_handler([_nonempty("ok")])

    response = mw.wrap_model_call(_FakeRequest(), handler)

    assert response is not None
    assert len(calls) == 1


def test_wrap_sync_empty_then_non_empty_retries() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = _sync_handler([_empty(), _nonempty("sync-recovered")])

    response = mw.wrap_model_call(_FakeRequest(), handler)

    assert _unwrap_ai_message(response).content == "sync-recovered"  # type: ignore[union-attr]
    assert len(calls) == 2


def test_wrap_sync_empty_always_returns_last_empty() -> None:
    mw = EmptyResponseRetryMiddleware(max_retries=1)
    handler, calls = _sync_handler([_empty(), _empty()])

    response = mw.wrap_model_call(_FakeRequest(), handler)

    assert _is_empty(_unwrap_ai_message(response))  # type: ignore[arg-type]
    assert len(calls) == 2


def test_wrap_sync_fallback_model_overrides_on_last_attempt() -> None:
    fallback = _FakeFallbackModel()
    mw = EmptyResponseRetryMiddleware(max_retries=1, fallback_model=fallback)
    request = _FakeRequest()
    handler, _ = _sync_handler([_empty(), _empty()])

    mw.wrap_model_call(request, handler)

    assert any(call.get("model") is fallback for call in request.override_calls)
