"""Unit tests for PromptCacheMiddleware."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from genai_tk.agents.langchain.middleware.prompt_cache import PromptCacheMiddleware


@dataclass
class FakeRequest:
    """Duck-typed stand-in for ModelRequest (only ``messages``/``override`` are used)."""

    messages: list
    override_kwargs: dict = field(default_factory=dict)

    def override(self, **kwargs):  # noqa: ANN003
        new = FakeRequest(messages=kwargs.get("messages", self.messages))
        new.override_kwargs = kwargs
        return new


def test_tags_leading_system_message():
    mw = PromptCacheMiddleware()
    request = FakeRequest(messages=[SystemMessage(content="sys"), HumanMessage(content="q")])
    captured = []
    result = mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    tagged = captured[0].messages[0]
    assert tagged.additional_kwargs["cache_control"] == {"type": "ephemeral"}
    assert captured[0].messages[1] is request.messages[1]
    assert result is not None


def test_tags_first_system_message_anywhere():
    mw = PromptCacheMiddleware()
    human, sys_a, sys_b = HumanMessage(content="q"), SystemMessage(content="a"), SystemMessage(content="b")
    request = FakeRequest(messages=[human, sys_a, sys_b])
    captured = []
    mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    assert "cache_control" not in human.additional_kwargs
    assert captured[0].messages[1].additional_kwargs["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in captured[0].messages[2].additional_kwargs


def test_no_double_tagging():
    mw = PromptCacheMiddleware()
    already = SystemMessage(content="sys", additional_kwargs={"cache_control": {"type": "ephemeral"}})
    request = FakeRequest(messages=[already])
    captured = []
    mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    assert captured[0].messages[0] is already  # untouched, no override


def test_no_system_message_is_noop():
    mw = PromptCacheMiddleware()
    request = FakeRequest(messages=[HumanMessage(content="q")])
    captured = []
    mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    assert captured[0] is request  # no override performed


def test_tag_all_system_messages():
    mw = PromptCacheMiddleware(tag_all_system=True)
    sys_a, sys_b = SystemMessage(content="a"), SystemMessage(content="b")
    request = FakeRequest(messages=[sys_a, sys_b])
    captured = []
    mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    assert captured[0].messages[0].additional_kwargs["cache_control"] == {"type": "ephemeral"}
    assert captured[0].messages[1].additional_kwargs["cache_control"] == {"type": "ephemeral"}


def test_custom_cache_control_payload():
    mw = PromptCacheMiddleware(cache_control={"type": "ephemeral", "ttl": "1h"})
    request = FakeRequest(messages=[SystemMessage(content="sys")])
    captured = []
    mw.wrap_model_call(request, lambda r: (captured.append(r), AIMessage(content="ok"))[1])
    assert captured[0].messages[0].additional_kwargs["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.asyncio
async def test_async_variant_tags():
    mw = PromptCacheMiddleware()
    request = FakeRequest(messages=[SystemMessage(content="sys")])
    captured = []

    async def handler(r):
        captured.append(r)
        return AIMessage(content="ok")

    await mw.awrap_model_call(request, handler)
    assert captured[0].messages[0].additional_kwargs["cache_control"] == {"type": "ephemeral"}
