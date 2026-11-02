"""Unit tests for the WrapUpMiddleware anti-give-up guardrail and cross-page limits."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from genai_tk.agents.langchain.middleware.wrap_up_middleware import WrapUpMiddleware


@dataclass
class FakeRequest:
    """Duck-typed stand-in for ModelRequest (only ``messages``/``override`` are used)."""

    messages: list
    override_kwargs: dict = field(default_factory=dict)

    def override(self, **kwargs):  # noqa: ANN003
        new = FakeRequest(messages=kwargs.get("messages", self.messages))
        new.override_kwargs = kwargs
        return new


def _tool_msg(section_ids: str = "s0") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_section_content", "args": {"section_ids": section_ids}, "id": "t", "type": "tool_call"}
        ],
    )


def _conversation(n_calls: int, distinct_sections: bool = False) -> list:
    msgs: list = [SystemMessage(content="sys"), HumanMessage(content="q")]
    for i in range(n_calls):
        section = f"s{i}" if distinct_sections else "s0"
        msgs.append(_tool_msg(section))
    return msgs


def _run(mw: WrapUpMiddleware, messages: list, responses: list[AIMessage]):
    """Drive wrap_model_call with scripted responses; return (final, captured requests)."""
    captured: list[FakeRequest] = []
    call_counter = {"n": 0}

    def handler(request: FakeRequest):
        captured.append(request)
        response = responses[min(call_counter["n"], len(responses) - 1)]
        call_counter["n"] += 1
        return response

    final = mw.wrap_model_call(FakeRequest(messages=messages), handler)
    return final, captured


def test_below_soft_limit_is_noop():
    mw = WrapUpMiddleware(soft_limit=4, hard_limit=6)
    request = FakeRequest(messages=_conversation(2))
    result = mw.wrap_model_call(request, lambda r: AIMessage(content="42"))
    assert result is not None
    assert len(request.messages) == 4  # untouched


def test_soft_limit_appends_nudge():
    mw = WrapUpMiddleware(soft_limit=4, hard_limit=6)
    _, captured = _run(mw, _conversation(4), [AIMessage(content="42")])
    injected = captured[0].messages[-1]
    assert isinstance(injected, SystemMessage)
    assert "WRAP UP" in injected.content
    assert captured[0].override_kwargs.get("tools", "kept") == "kept"  # tools intact


def test_hard_limit_strips_tools():
    mw = WrapUpMiddleware(soft_limit=4, hard_limit=6)
    _, captured = _run(mw, _conversation(6), [AIMessage(content="42")])
    assert captured[0].override_kwargs["tools"] == []
    assert "tool budget is exhausted" in captured[0].messages[-1].content
    assert "Not answerable" in captured[0].messages[-1].content


def test_cross_page_conversation_extends_limits():
    mw = WrapUpMiddleware(soft_limit=4, hard_limit=6)  # extended defaults: 12 / 14
    _, captured = _run(mw, _conversation(6, distinct_sections=True), [AIMessage(content="42")])
    # 6 distinct sections >= 3 → cross-page → 6 < extended soft limit 12 → no nudge.
    assert len(captured[0].messages) == 8  # unchanged


def test_give_up_after_nudge_triggers_one_recovery():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(mw, _conversation(2), [AIMessage(content="I cannot find the answer.")])
    assert len(captured) == 2
    recovery = captured[1].messages[-1]
    assert isinstance(recovery, SystemMessage)
    assert "MANDATORY RECOVERY" in recovery.content
    assert "Not answerable" in recovery.content


def test_give_up_recovery_fires_only_once():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    give_up = AIMessage(content="Unable to locate the information.")
    first = mw.wrap_model_call(FakeRequest(messages=_conversation(2)), lambda r: give_up)
    assert first is give_up  # recovery response returned as-is after retry
    second_calls = []
    mw.wrap_model_call(FakeRequest(messages=_conversation(3)), lambda r: second_calls.append(1))
    assert len(second_calls) == 1  # guardrail spent, subsequent give-ups pass through


def test_legit_not_answerable_never_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(mw, _conversation(2), [AIMessage(content="Not answerable.")])
    assert len(captured) == 1


def test_legit_does_not_provide_never_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(mw, _conversation(2), [AIMessage(content="The document does not provide information about X.")])
    assert len(captured) == 1


def test_empty_final_after_nudge_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(mw, _conversation(2), [AIMessage(content="")])
    assert len(captured) == 2


def test_no_recovery_below_soft_limit():
    mw = WrapUpMiddleware(soft_limit=8, hard_limit=100)
    _, captured = _run(mw, _conversation(2), [AIMessage(content="I cannot find the answer.")])
    assert len(captured) == 1


def test_tool_call_response_never_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    still_working = AIMessage(
        content="",
        tool_calls=[{"name": "get_section_content", "args": {"section_ids": "s1"}, "id": "t", "type": "tool_call"}],
    )
    _, captured = _run(mw, _conversation(2), [still_working])
    assert len(captured) == 1


def test_unfinished_narration_below_soft_limit_recovers():
    """A narration that promises more work is not a final answer, even early on."""
    mw = WrapUpMiddleware(soft_limit=16, hard_limit=24)
    narration = AIMessage(
        content=(
            "I have the AY-1 table but the year headers were lost in extraction. "
            "Let me cross-check the year mapping using other bulletins' yield tables and "
            "search for the Railroad Retirement trust receipts table."
        )
    )
    _, captured = _run(mw, _conversation(3), [narration])
    assert len(captured) == 2
    recovery = captured[1].messages[-1]
    assert "MANDATORY RECOVERY" in recovery.content
    assert "without delivering a final answer" in recovery.content


def test_real_answer_with_high_tool_count_never_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(
        mw,
        _conversation(3),
        [AIMessage(content="The total was 92,000,000 dollars (Table AY-1, June 1970 bulletin).")],
    )
    assert len(captured) == 1


def test_long_response_is_never_treated_as_narration():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    long_text = "Analysis: " + ("x" * 900) + " Let me check further."
    _, captured = _run(mw, _conversation(3), [AIMessage(content=long_text)])
    assert len(captured) == 1


def test_legit_unanswerable_narration_never_recovers():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    _, captured = _run(
        mw,
        _conversation(3),
        [AIMessage(content="The document does not provide the figure, so I cannot report one.")],
    )
    assert len(captured) == 1


def test_narration_recovery_fires_only_once():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    narration = AIMessage(content="Let me check the next table.")
    mw.wrap_model_call(FakeRequest(messages=_conversation(3)), lambda r: narration)
    calls: list[int] = []
    mw.wrap_model_call(FakeRequest(messages=_conversation(4)), lambda r: calls.append(1))
    assert len(calls) == 1  # guardrail spent


async def test_async_give_up_recovery():
    mw = WrapUpMiddleware(soft_limit=2, hard_limit=100)
    captured: list[FakeRequest] = []
    responses = iter([AIMessage(content="I could not find it."), AIMessage(content="42")])

    async def handler(request: FakeRequest):
        captured.append(request)
        return next(responses)

    result = await mw.awrap_model_call(FakeRequest(messages=_conversation(2)), handler)
    assert len(captured) == 2
    assert "MANDATORY RECOVERY" in captured[1].messages[-1].content
    assert result is not None
