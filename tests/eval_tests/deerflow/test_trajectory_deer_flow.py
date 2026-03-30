"""Trajectory evaluation tests for Deer-flow agents.

Unlike the LangchainAgent trajectory tests, deer-flow does not support fake
LLM injection — all tests here use a real LLM and are gated behind both
``--include-real-models`` and ``DEER_FLOW_PATH``.

Test categories
---------------
- **Structure tests** — Verify the trajectory format emitted by
  ``DeerFlowEvalClient.acollect_trajectory`` is well-formed (correct roles,
  tool-call / tool-result pairing).  These validate the harness itself.
- **Tool-presence tests** — Verify the agent triggers tool calls for queries
  that require computation or code execution.
- **agentevals match tests** — Use ``create_trajectory_match_evaluator`` to
  assert that expected tool names appear in the actual trajectory.

All tests are ``@pytest.mark.real_models`` and ``@pytest.mark.deerflow``.

Design note: tool names are *not* pinned to exact deer-flow internals wherever
avoidable.  Instead, structural assertions (``"tool_calls" in message``) and
superset matching are preferred so tests remain valid across deer-flow versions.
"""

from __future__ import annotations

import asyncio

import pytest

AGENT_TIMEOUT = 180  # seconds — deer-flow may think / plan before responding


# ---------------------------------------------------------------------------
# Trajectory structure tests  (validate the harness format)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_trajectory_starts_with_user_message(deerflow_client) -> None:
    """The collected trajectory always begins with the user message."""
    trajectory = asyncio.run(deerflow_client.acollect_trajectory("What is 2 + 2?"))
    assert trajectory, "Trajectory must not be empty"
    assert trajectory[0]["role"] == "user", f"First message must have role 'user', got {trajectory[0]['role']!r}"
    assert trajectory[0]["content"] == "What is 2 + 2?"


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_trajectory_ends_with_assistant_message(deerflow_client) -> None:
    """The last message in the trajectory is an assistant text response."""
    trajectory = asyncio.run(deerflow_client.acollect_trajectory("What is the capital of France?"))
    assert any(m["role"] == "assistant" for m in trajectory), "Trajectory must contain at least one assistant message"
    final = next(m for m in reversed(trajectory) if m["role"] == "assistant")
    assert final.get("content"), f"Final assistant message must have non-empty content.\nFull trajectory: {trajectory}"


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_tool_result_follows_tool_call(deerflow_client) -> None:
    """Every tool call in the trajectory is followed by a tool result message.

    If the agent calls no tools for this query the test is skipped (not failed)
    — we are testing structure validity, not forcing tool use.
    """
    trajectory = asyncio.run(
        deerflow_client.acollect_trajectory("Execute this Python and show me the output: result = 6 * 7; print(result)")
    )

    tool_call_indices = [i for i, m in enumerate(trajectory) if m.get("tool_calls")]
    if not tool_call_indices:
        pytest.skip("Agent answered this query without calling any tools — skipping structure check")

    for tc_idx in tool_call_indices:
        remaining = trajectory[tc_idx + 1 :]
        assert any(m["role"] == "tool" for m in remaining), (
            f"Expected a tool result after tool_call at index {tc_idx}.\nFull trajectory: {trajectory}"
        )


# ---------------------------------------------------------------------------
# Tool-presence tests  (verify the agent uses tools when appropriate)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_tool_called_for_explicit_code_execution_request(deerflow_client) -> None:
    """A query explicitly requesting code execution triggers at least one tool call.

    The query is phrased to strongly encourage tool use so the agent does not
    answer from memory alone (which would be trivially correct but miss the
    point of using a code execution tool).
    """
    trajectory = asyncio.run(
        deerflow_client.acollect_trajectory(
            "Run Python code to compute sum(range(1, 11)) and tell me the answer. "
            "Do not guess — actually execute the code."
        )
    )

    tool_call_messages = [m for m in trajectory if m.get("tool_calls")]
    assert tool_call_messages, (
        f"Expected at least one tool call for an explicit code-execution request.\nFull trajectory: {trajectory}"
    )


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_tool_call_has_valid_structure(deerflow_client) -> None:
    """Tool-call entries in the trajectory conform to the OpenAI function-call schema."""
    trajectory = asyncio.run(deerflow_client.acollect_trajectory("Run Python to evaluate: 2 ** 10. Show the result."))

    tool_call_messages = [m for m in trajectory if m.get("tool_calls")]
    if not tool_call_messages:
        pytest.skip("Agent answered without calling tools — skipping structure validation")

    for msg in tool_call_messages:
        assert msg["role"] == "assistant", "Tool-call message must have role 'assistant'"
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            assert fn.get("name"), f"tool_call.function.name must be non-empty: {tc}"
            assert "arguments" in fn, f"tool_call.function.arguments must be present: {tc}"


# ---------------------------------------------------------------------------
# agentevals trajectory match
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(AGENT_TIMEOUT)
def test_trajectory_match_superset_for_code_execution(deerflow_client) -> None:
    """agentevals superset evaluator passes when a tool call is present.

    The reference trajectory specifies that *some* assistant tool-call turn
    must exist.  The actual tool name is intentionally left as a wildcard
    (``tool_args_match_mode="ignore"``, ``trajectory_match_mode="superset"``)
    so the test is not brittle to deer-flow's internal tool naming.
    """
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    actual_trajectory = asyncio.run(
        deerflow_client.acollect_trajectory("Execute Python: import math; print(math.factorial(5)). Show the output.")
    )

    has_tool_call = any(m.get("tool_calls") for m in actual_trajectory)
    if not has_tool_call:
        pytest.skip("Agent answered without tool calls — superset match would be trivially false")

    # Extract the actual tool name from the trajectory so the evaluator can
    # match it exactly.  This tests that our trajectory format is compatible
    # with agentevals — not which specific tool was selected.
    actual_tool_name = next(
        tc["function"]["name"]
        for m in actual_trajectory
        if m.get("tool_calls")
        for tc in m["tool_calls"]
        if tc.get("function", {}).get("name")
    )

    reference_trajectory = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": actual_tool_name, "arguments": "{}"}}],
        }
    ]

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    result = evaluator(outputs=actual_trajectory, reference_outputs=reference_trajectory)
    assert result["score"], (
        f"Trajectory superset match failed.\nFeedback: {result.get('feedback')}\nActual trajectory: {actual_trajectory}"
    )
