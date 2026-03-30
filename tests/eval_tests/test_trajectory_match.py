"""Deterministic trajectory match evaluation tests.

These tests validate that:
1. ``create_trajectory_match_evaluator`` from agentevals correctly evaluates
   pre-built reference trajectories (no LLM required, pure deterministic logic).
2. ``extract_message_trajectory`` produces a well-formed OpenAI-format message
   list when run against a mocked compiled LangGraph (no real LLM invocation).
3. Real-model integration: an actual React agent with tools calls the expected
   tool and its trajectory passes a ``superset`` evaluation (opt-in, gated by
   ``--include-real-models``).

All pure-evaluator tests (classes ``TestSuperset*``, ``TestSubset*``,
``TestStrict*``) have zero dependencies on external services.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.eval_tests.helpers import (
    assert_tool_not_called,
    assert_tool_was_called,
    get_final_response,
    make_assistant_message,
    make_tool_call,
    make_tool_message,
    make_user_message,
    tool_names_in_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures — pre-built trajectories
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator_trajectory() -> list[dict]:
    """A realistic trajectory showing the agent calling the calculator tool."""
    return [
        make_user_message("What is 17 * 3?"),
        make_assistant_message(tool_calls=[make_tool_call("calculator", {"expression": "17 * 3"})]),
        make_tool_message("51.0"),
        make_assistant_message("17 * 3 equals 51."),
    ]


@pytest.fixture
def echo_trajectory() -> list[dict]:
    """A realistic trajectory showing the agent calling the echo tool."""
    return [
        make_user_message("Echo back 'hello world'"),
        make_assistant_message(tool_calls=[make_tool_call("echo", {"message": "hello world"})]),
        make_tool_message("hello world"),
        make_assistant_message("The echo result is: hello world"),
    ]


@pytest.fixture
def both_tools_trajectory() -> list[dict]:
    """A trajectory where the agent calls both calculator and echo."""
    return [
        make_user_message("Calculate 5+5 then echo 'done'"),
        make_assistant_message(tool_calls=[make_tool_call("calculator", {"expression": "5+5"})]),
        make_tool_message("10.0"),
        make_assistant_message(tool_calls=[make_tool_call("echo", {"message": "done"})]),
        make_tool_message("done"),
        make_assistant_message("Calculated 10 and echoed 'done'."),
    ]


@pytest.fixture
def no_tool_trajectory() -> list[dict]:
    """A trajectory where the agent answers directly without tools."""
    return [
        make_user_message("What colour is the sky?"),
        make_assistant_message("The sky is blue."),
    ]


# ---------------------------------------------------------------------------
# Helpers assertion tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_tool_names_in_trajectory(self, calculator_trajectory) -> None:
        names = tool_names_in_trajectory(calculator_trajectory)
        assert names == ["calculator"]

    def test_tool_names_both(self, both_tools_trajectory) -> None:
        names = tool_names_in_trajectory(both_tools_trajectory)
        assert names == ["calculator", "echo"]

    def test_tool_names_empty(self, no_tool_trajectory) -> None:
        assert tool_names_in_trajectory(no_tool_trajectory) == []

    def test_assert_tool_was_called_passes(self, calculator_trajectory) -> None:
        assert_tool_was_called(calculator_trajectory, "calculator")  # must not raise

    def test_assert_tool_was_called_fails(self, no_tool_trajectory) -> None:
        with pytest.raises(AssertionError, match="calculator"):
            assert_tool_was_called(no_tool_trajectory, "calculator")

    def test_assert_tool_not_called_passes(self, calculator_trajectory) -> None:
        assert_tool_not_called(calculator_trajectory, "echo")  # must not raise

    def test_assert_tool_not_called_fails(self, calculator_trajectory) -> None:
        with pytest.raises(AssertionError, match="calculator"):
            assert_tool_not_called(calculator_trajectory, "calculator")

    def test_get_final_response(self, calculator_trajectory) -> None:
        assert get_final_response(calculator_trajectory) == "17 * 3 equals 51."

    def test_get_final_response_no_assistant(self, no_tool_trajectory) -> None:
        # Last assistant message in no-tool trajectory
        assert get_final_response(no_tool_trajectory) == "The sky is blue."


# ---------------------------------------------------------------------------
# Superset mode — agent called at LEAST the expected tools
# ---------------------------------------------------------------------------


@pytest.mark.evals
class TestSupersetMatch:
    """Agent trajectory must contain at least the expected tool calls."""

    def _evaluator(self):
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        return create_trajectory_match_evaluator(
            trajectory_match_mode="superset",
            tool_args_match_mode="ignore",
        )

    def test_exact_tool_call_passes(self, calculator_trajectory) -> None:
        """Reference tool matches what agent called → superset passes."""
        reference = [
            make_user_message("What is 17 * 3?"),
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
        ]
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=reference)
        assert result["score"] is True

    def test_multiple_tools_superset_passes(self, both_tools_trajectory) -> None:
        """Agent called both tools — superset of reference with one tool — passes."""
        reference = [
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
        ]
        result = self._evaluator()(outputs=both_tools_trajectory, reference_outputs=reference)
        assert result["score"] is True

    def test_missing_expected_tool_fails(self, no_tool_trajectory) -> None:
        """Agent did NOT call calculator, but reference requires it → fails."""
        reference = [
            make_user_message("What is 17 * 3?"),
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
        ]
        result = self._evaluator()(outputs=no_tool_trajectory, reference_outputs=reference)
        assert result["score"] is False

    def test_wrong_tool_fails(self, echo_trajectory) -> None:
        """Agent called echo instead of calculator → fails for calculator reference."""
        reference = [
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
        ]
        result = self._evaluator()(outputs=echo_trajectory, reference_outputs=reference)
        assert result["score"] is False

    def test_feedback_key_in_result(self, calculator_trajectory) -> None:
        """EvaluatorResult contains the expected 'key' field."""
        reference = [make_assistant_message(tool_calls=[make_tool_call("calculator")])]
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=reference)
        assert "trajectory_superset_match" in result["key"]


# ---------------------------------------------------------------------------
# Subset mode — agent called ONLY tools that appear in reference
# ---------------------------------------------------------------------------


@pytest.mark.evals
class TestSubsetMatch:
    """Agent trajectory tool calls must all be present in the reference."""

    def _evaluator(self):
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        return create_trajectory_match_evaluator(
            trajectory_match_mode="subset",
            tool_args_match_mode="ignore",
        )

    def test_subset_passes_when_agent_calls_fewer(self, calculator_trajectory) -> None:
        """Agent called only calculator, reference has calculator+echo → passes."""
        reference = [
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
            make_assistant_message(tool_calls=[make_tool_call("echo")]),
        ]
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=reference)
        assert result["score"] is True

    def test_subset_passes_exact_match(self, calculator_trajectory) -> None:
        """Exact match is also a valid subset."""
        reference = [
            make_user_message("What is 17 * 3?"),
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
            make_tool_message("51.0"),
            make_assistant_message("17 * 3 equals 51."),
        ]
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=reference)
        assert result["score"] is True

    def test_subset_fails_when_agent_calls_extra_tool(self, both_tools_trajectory) -> None:
        """Agent called calculator AND echo; reference only has calculator → fails."""
        reference = [
            make_assistant_message(tool_calls=[make_tool_call("calculator")]),
        ]
        result = self._evaluator()(outputs=both_tools_trajectory, reference_outputs=reference)
        assert result["score"] is False


# ---------------------------------------------------------------------------
# Strict mode — trajectory must match exactly
# ---------------------------------------------------------------------------


@pytest.mark.evals
class TestStrictMatch:
    """Agent trajectory must match reference exactly."""

    def _evaluator(self):
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        return create_trajectory_match_evaluator(
            trajectory_match_mode="strict",
            tool_args_match_mode="ignore",
        )

    def test_exact_trajectory_passes(self, calculator_trajectory) -> None:
        """Identical trajectories pass strict match."""
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=calculator_trajectory)
        assert result["score"] is True

    def test_different_tool_fails(self, calculator_trajectory, echo_trajectory) -> None:
        """Different tool names → strict match fails."""
        result = self._evaluator()(outputs=calculator_trajectory, reference_outputs=echo_trajectory)
        assert result["score"] is False

    def test_extra_messages_fail(self, both_tools_trajectory, calculator_trajectory) -> None:
        """Agent called extra tools → strict match fails."""
        result = self._evaluator()(outputs=both_tools_trajectory, reference_outputs=calculator_trajectory)
        assert result["score"] is False


# ---------------------------------------------------------------------------
# Trajectory extraction from a mocked LangGraph (no real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.asyncio
async def test_extract_message_trajectory_format() -> None:
    """extract_message_trajectory returns valid OpenAI-format dicts from mock graph."""

    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory

    # Build a realistic LangGraph state with a tool call sequence
    human = HumanMessage(content="What is 6 * 7?")
    ai_with_tool = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "6 * 7"}, "id": "tc1", "type": "tool_call"}],
    )
    tool_result = ToolMessage(content="42.0", tool_call_id="tc1")
    final = AIMessage(content="6 * 7 equals 42.")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [human, ai_with_tool, tool_result, final]}

    agent = LangchainAgent("eval-test", llm="parrot_local@fake")
    agent._agent = mock_graph  # Inject mock to skip real compilation

    trajectory = await extract_message_trajectory(agent, "What is 6 * 7?")

    # Check structure
    assert len(trajectory) == 4
    roles = [m["role"] for m in trajectory]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # Check tool_calls were extracted
    ai_msg = trajectory[1]
    assert "tool_calls" in ai_msg
    assert ai_msg["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(ai_msg["tool_calls"][0]["function"]["arguments"]) == {"expression": "6 * 7"}

    # Final assistant message has no tool_calls
    final_msg = trajectory[3]
    assert "tool_calls" not in final_msg
    assert final_msg["content"] == "6 * 7 equals 42."


@pytest.mark.evals
@pytest.mark.asyncio
async def test_extract_message_trajectory_passes_superset_eval() -> None:
    """Extracted trajectory from mock graph passes superset evaluator for calculator call."""

    from agentevals.trajectory.match import create_trajectory_match_evaluator

    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory

    # Pre-build the mock state
    human = HumanMessage(content="What is 6 * 7?")
    ai_with_tool = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "6 * 7"}, "id": "tc1", "type": "tool_call"}],
    )
    tool_result = ToolMessage(content="42.0", tool_call_id="tc1")
    final = AIMessage(content="6 * 7 equals 42.")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [human, ai_with_tool, tool_result, final]}

    agent = LangchainAgent("eval-test", llm="parrot_local@fake")
    agent._agent = mock_graph

    trajectory = await extract_message_trajectory(agent, "What is 6 * 7?")

    # Verify trajectory against a reference that requires calculator to have been called
    reference = [
        make_user_message("What is 6 * 7?"),
        make_assistant_message(tool_calls=[make_tool_call("calculator")]),
    ]
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    result = evaluator(outputs=trajectory, reference_outputs=reference)
    assert result["score"] is True, f"Superset evaluation failed: {result}"


# ---------------------------------------------------------------------------
# Graph-level trajectory — strict node-step matching (mocked, no real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.evals
def test_graph_trajectory_strict_match_pass() -> None:
    """graph_trajectory_strict_match passes when node sequences match exactly."""
    from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

    reference = {"steps": ["__start__", "agent", "tools", "agent"]}
    outputs = {"steps": ["__start__", "agent", "tools", "agent"]}
    result = graph_trajectory_strict_match(outputs=outputs, reference_outputs=reference)
    assert result["score"] is True


@pytest.mark.evals
def test_graph_trajectory_strict_match_fail() -> None:
    """graph_trajectory_strict_match fails when node sequences differ."""
    from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

    reference = {"steps": ["__start__", "agent", "tools", "agent"]}
    outputs = {"steps": ["__start__", "agent"]}  # Agent didn't use tools
    result = graph_trajectory_strict_match(outputs=outputs, reference_outputs=reference)
    assert result["score"] is False


# ---------------------------------------------------------------------------
# Real-model integration test (opt-in, --include-real-models)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.asyncio
async def test_react_agent_calls_calculator_with_real_model(eval_agent) -> None:
    """Real React agent calls the calculator tool on an arithmetic query.

    The trajectory must pass a superset check confirming the agent used the
    calculator tool at least once.

    Requires: ``--include-real-models`` AND a valid API key in the environment.
    """
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    # Override the LLM to use the fast / cheap model for evals
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory
    from tests.eval_tests.conftest import calculator, echo

    agent = LangchainAgent("eval-test", llm="fast_model", agent_type="react", tools=[calculator, echo])
    trajectory = await extract_message_trajectory(agent, "What is 23 * 7? Use the calculator tool.")

    reference = [
        make_user_message("What is 23 * 7? Use the calculator tool."),
        make_assistant_message(tool_calls=[make_tool_call("calculator")]),
    ]
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    result = evaluator(outputs=trajectory, reference_outputs=reference)
    assert result["score"] is True, "Agent did not call calculator. Trajectory:\n" + "\n".join(
        f"  {m['role']}: {str(m)[:120]}" for m in trajectory
    )
