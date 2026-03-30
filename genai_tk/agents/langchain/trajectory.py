"""Trajectory extraction utilities for LangChain/LangGraph agents.

Converts LangGraph state messages into standardised OpenAI-format dicts that
are natively accepted by both ``openevals`` and ``agentevals`` evaluators.

Usage:
    ```python
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory

    agent = LangchainAgent(llm="fast_model", agent_type="react", tools=[my_tool])
    trajectory = await extract_message_trajectory(agent, "What is the capital of France?")

    # trajectory is a list of OpenAI-style message dicts, e.g.:
    # [
    #   {"role": "user", "content": "What is the capital of France?"},
    #   {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "..."}'}}]},
    #   {"role": "tool",      "content": "Paris is the capital of France."},
    #   {"role": "assistant", "content": "The capital of France is Paris."},
    # ]
    ```
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent


def messages_to_openai_format(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of LangChain ``BaseMessage`` objects to OpenAI-format dicts.

    Both ``openevals`` and ``agentevals`` evaluators accept LangChain
    ``BaseMessage`` instances directly, but having a plain-dict representation
    makes tests easier to write reference trajectories for.

    Args:
        messages: List of LangChain ``BaseMessage`` objects (or already-formatted dicts).

    Returns:
        List of dicts with at least ``role`` and ``content`` keys.
        Assistant messages with tool calls include a ``tool_calls`` list.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        # Already a plain dict — pass through unchanged
        if isinstance(msg, dict):
            result.append(msg)
            continue

        role = _get_role(msg)
        content = _get_content(msg)
        entry: dict[str, Any] = {"role": role, "content": content}

        # Attach tool_calls for assistant messages that triggered tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                        "arguments": _serialise_args(
                            tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        ),
                    }
                }
                for tc in tool_calls
            ]

        result.append(entry)
    return result


async def extract_message_trajectory(
    agent: LangchainAgent,
    query: str,
    *,
    thread_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``agent`` on ``query`` and return the full message trajectory.

    The trajectory includes the user message, all intermediate tool calls and
    their results, and the final assistant response — in chronological order.

    Args:
        agent: A ``LangchainAgent`` instance (will be lazily initialised).
        query: The user query to execute.
        thread_id: Optional thread ID for stateful (checkpointer) agents.
            Generates a UUID if omitted.

    Returns:
        OpenAI-format message list representing the full execution trajectory.
    """
    compiled = await agent._ensure_initialized()
    tid = thread_id or str(uuid.uuid4())

    result = await compiled.ainvoke(
        {"messages": query},
        {"configurable": {"thread_id": tid}},
    )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    return messages_to_openai_format(messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_role(msg: Any) -> str:
    """Map a LangChain message class to an OpenAI role string."""
    class_name = type(msg).__name__
    if class_name in ("HumanMessage", "HumanMessageChunk"):
        return "user"
    if class_name in ("AIMessage", "AIMessageChunk"):
        return "assistant"
    if class_name == "ToolMessage":
        return "tool"
    if class_name == "SystemMessage":
        return "system"
    # Generic fallback — use the ``type`` attribute if present
    return getattr(msg, "type", "assistant")


def _get_content(msg: Any) -> str:
    """Extract plain-string content from a message."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        # Multi-part content (e.g. vision) — join text parts
        return "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content) if content is not None else ""


def _serialise_args(args: Any) -> str:
    """Serialise tool call arguments to a JSON string."""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args)
    except (TypeError, ValueError):
        return str(args)
