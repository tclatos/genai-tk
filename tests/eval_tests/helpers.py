"""Shared helpers for agent evaluation tests.

Provides convenience wrappers around ``genai_tk.agents.langchain.trajectory``
and reference-trajectory construction helpers used by the evaluation test
modules.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------


async def run_agent_and_capture_trajectory(
    agent: Any,
    query: str,
    *,
    thread_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``agent`` on ``query`` and return the full OpenAI-format trajectory.

    Args:
        agent: A ``LangchainAgent`` instance.
        query: Query string to run.
        thread_id: Optional thread ID for checkpointer-backed agents.

    Returns:
        List of OpenAI-format message dicts (role, content, optional tool_calls).
    """
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory

    return await extract_message_trajectory(agent, query, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Reference trajectory construction helpers
# ---------------------------------------------------------------------------


def make_user_message(content: str) -> dict[str, Any]:
    """Build an OpenAI-format ``user`` message dict.

    Args:
        content: The user message text.
    """
    return {"role": "user", "content": content}


def make_assistant_message(content: str = "", tool_calls: list[dict] | None = None) -> dict[str, Any]:
    """Build an OpenAI-format ``assistant`` message dict.

    Args:
        content: Optional text content (typically empty when tool calls are present).
        tool_calls: Optional list of tool call dicts in OpenAI format.
    """
    entry: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


def make_tool_call(tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an OpenAI-format tool call dict for use inside ``tool_calls``.

    Args:
        tool_name: The tool function name.
        args: Arguments passed to the tool (defaults to empty dict).
    """
    return {
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args or {}),
        }
    }


def make_tool_message(content: str) -> dict[str, Any]:
    """Build an OpenAI-format ``tool`` result message dict.

    Args:
        content: The tool response content.
    """
    return {"role": "tool", "content": content}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def tool_names_in_trajectory(trajectory: list[dict[str, Any]]) -> list[str]:
    """Return all tool names that were called in ``trajectory``, in order.

    Args:
        trajectory: OpenAI-format message list from ``extract_message_trajectory``.

    Returns:
        List of tool names that appeared in assistant ``tool_calls``.
    """
    names: list[str] = []
    for msg in trajectory:
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name:
                names.append(name)
    return names


def assert_tool_was_called(trajectory: list[dict[str, Any]], tool_name: str) -> None:
    """Assert that ``tool_name`` appears at least once in ``trajectory``.

    Args:
        trajectory: OpenAI-format message list.
        tool_name: The expected tool function name.

    Raises:
        AssertionError: With a descriptive message if the tool was not called.
    """
    called = tool_names_in_trajectory(trajectory)
    assert tool_name in called, f"Expected tool '{tool_name}' to be called, but trajectory only contains: {called}"


def assert_tool_not_called(trajectory: list[dict[str, Any]], tool_name: str) -> None:
    """Assert that ``tool_name`` was NOT called in ``trajectory``.

    Args:
        trajectory: OpenAI-format message list.
        tool_name: The tool name that should not appear.

    Raises:
        AssertionError: If the tool was called.
    """
    called = tool_names_in_trajectory(trajectory)
    assert tool_name not in called, (
        f"Expected tool '{tool_name}' NOT to be called, but it appeared in trajectory: {called}"
    )


def get_final_response(trajectory: list[dict[str, Any]]) -> str:
    """Return the content of the last assistant message in ``trajectory``.

    Args:
        trajectory: OpenAI-format message list.

    Returns:
        Content string of the last assistant message, or empty string.
    """
    for msg in reversed(trajectory):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""
