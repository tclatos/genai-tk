"""Generic content block extraction and message decomposition utilities.

This module provides normalized access to modern LangChain ``content_blocks``
(introduced in LangChain 1.6+) across standard messages (``AIMessage``,
``AIMessageChunk``, ``HumanMessage``, ``ToolMessage``), model response wrappers
(DeepAgents ``ModelResponse``, ``ExtendedModelResponse``), and raw dict/string payloads.

It decomposes any message or streaming chunk into:
- ``text``: User-facing visible text from ``{"type": "text"}`` blocks.
- ``thinking``: Chain-of-thought / reasoning traces from ``{"type": "reasoning"}``,
  ``{"type": "thought"}``, ``{"type": "thinking"}``, or ``additional_kwargs`` reasoning fields.
- ``tool_calls``: Standard tool call invocations.
- ``blocks``: Raw normalized content blocks.

Also provides tag sanitization for models that leak raw markdown reasoning markers
(``<think>...</think>``, ``assistantfinal``, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Matches <think>...</think>, <reasoning>...</reasoning>, <thought>...</thought>, <thinking>...</thinking>
_THINKING_TAG_RE = re.compile(
    r"<(think|thinking|thought|reasoning)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# Matches unclosed opening thinking tags at the start: e.g. <think> ... (when truncated)
_UNCLOSED_THINKING_TAG_RE = re.compile(
    r"^\s*<(think|thinking|thought|reasoning)[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)
# Matches DeerFlow / model-generated assistantfinal markers
_ASSISTANT_FINAL_RE = re.compile(r"assistantfinal", re.IGNORECASE)


class AIMessageParts(BaseModel):
    """Decomposed parts of an AI message or chunk."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    blocks: list[dict[str, Any]] = Field(default_factory=list)


def unwrap_message(obj: Any) -> Any:
    """Unwrap DeepAgents ModelResponse, ExtendedModelResponse, or list wrappers."""
    inner = obj
    if hasattr(inner, "model_response"):
        inner = inner.model_response
    if hasattr(inner, "result"):
        msgs = inner.result
        if msgs and isinstance(msgs, list):
            inner = msgs[0]
    return inner


def extract_content_blocks(message: Any) -> list[dict[str, Any]]:
    """Extract standard, normalized content block dicts from any message or chunk.

    Uses LangChain's ``BaseMessage.content_blocks`` property when available, and
    supplements it with any non-standard reasoning fields in ``additional_kwargs``
    or ``non_standard`` blocks (e.g. ``thought``, ``thinking``, ``reasoning``).

    Args:
        message: An ``AIMessage``, ``AIMessageChunk``, wrapper object, dict, or string.

    Returns:
        List of normalized block dicts.
    """
    msg = unwrap_message(message)
    if msg is None:
        return []

    blocks: list[dict[str, Any]] = []

    # 1. Try LangChain 1.6+ content_blocks property
    raw_blocks = getattr(msg, "content_blocks", None)
    if raw_blocks is not None and isinstance(raw_blocks, list):
        for b in raw_blocks:
            if isinstance(b, dict):
                blocks.append(dict(b))
            else:
                blocks.append({"type": "text", "text": str(b)})

    # 2. If content_blocks wasn't available or returned empty, inspect .content directly
    if not blocks:
        content = getattr(msg, "content", msg if isinstance(msg, (str, list)) else "")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    b_type = part.get("type", "text")
                    if b_type in ("text", "reasoning", "thought", "thinking", "tool_call"):
                        blocks.append(dict(part))
                    elif "text" in part:
                        blocks.append({"type": "text", "text": part["text"]})
                    elif "reasoning" in part:
                        blocks.append({"type": "reasoning", "reasoning": part["reasoning"]})
                    elif "thought" in part:
                        blocks.append({"type": "thought", "thought": part["thought"]})
                    elif "thinking" in part:
                        blocks.append({"type": "thinking", "thinking": part["thinking"]})
                    else:
                        blocks.append({"type": "non_standard", "value": part})
                elif part:
                    blocks.append({"type": "text", "text": str(part)})

    # 3. Supplemental check for reasoning/thought in additional_kwargs if not already captured
    has_reasoning = any(b.get("type") in ("reasoning", "thought", "thinking") for b in blocks)
    additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
    if not has_reasoning and isinstance(additional_kwargs, dict):
        for key in ("reasoning_content", "reasoning", "thought", "thinking"):
            val = additional_kwargs.get(key)
            if val and isinstance(val, str):
                blocks.insert(0, {"type": "reasoning", "reasoning": val})
                break

    # 4. Include tool_calls if present on the message and missing from blocks
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        existing_tc_ids = {b.get("id") for b in blocks if b.get("type") == "tool_call" and b.get("id")}
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            if not tc_id or tc_id not in existing_tc_ids:
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": tc_id,
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }
                )

    return blocks


def extract_ai_message_parts(message: Any) -> AIMessageParts:
    """Decompose an AI message or chunk into its text, thinking, and tool calls.

    Args:
        message: An ``AIMessage``, ``AIMessageChunk``, wrapper object, dict, or string.

    Returns:
        Structured :class:`AIMessageParts` instance.
    """
    blocks = extract_content_blocks(message)
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        b_type = block.get("type", "")
        if b_type == "text":
            text_parts.append(block.get("text", ""))
        elif b_type in ("reasoning", "thought", "thinking"):
            # Normalize reasoning text from any of the standard/alternative field names
            thought_text = (
                block.get("reasoning") or block.get("thought") or block.get("thinking") or block.get("text") or ""
            )
            thinking_parts.append(thought_text)
        elif b_type == "tool_call":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "args": block.get("args", {}),
                }
            )
        elif b_type == "non_standard":
            val = block.get("value")
            if isinstance(val, dict):
                sub_type = val.get("type", "")
                if sub_type in ("reasoning", "thought", "thinking"):
                    t_text = val.get("reasoning") or val.get("thought") or val.get("thinking") or val.get("text") or ""
                    thinking_parts.append(t_text)
                elif "text" in val:
                    text_parts.append(val["text"])
                elif "thought" in val:
                    thinking_parts.append(val["thought"])
                elif "reasoning" in val:
                    thinking_parts.append(val["reasoning"])

    # If no blocks were classified as text but msg had raw content string with no reasoning
    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts)

    return AIMessageParts(
        text=full_text,
        thinking=full_thinking,
        tool_calls=tool_calls,
        blocks=blocks,
    )


def extract_text_content(message: Any) -> str:
    """Extract only the user-facing text content from a message, skipping reasoning."""
    return extract_ai_message_parts(message).text


def extract_thinking_content(message: Any) -> str:
    """Extract only the thinking/reasoning trace from a message."""
    return extract_ai_message_parts(message).thinking


def strip_reasoning_tags(text: str) -> str:
    """Strip inline reasoning tags (<think>...</think>, assistantfinal, etc.) from raw text.

    Args:
        text: Raw text string that may contain leaked reasoning tags.

    Returns:
        Cleaned string with reasoning sections removed.
    """
    if not text:
        return ""

    # 1. Strip assistantfinal markers if present
    matches = list(_ASSISTANT_FINAL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end() :]

    # 2. Strip closed thinking tags <think>...</think>
    text = _THINKING_TAG_RE.sub("", text)

    # 3. If an unclosed thinking tag remains at the start (truncated thinking output), strip it
    # Only if there's no corresponding closing tag
    if _UNCLOSED_THINKING_TAG_RE.match(text) and not any(
        tag in text for tag in ("</think>", "</thinking>", "</thought>", "</reasoning>")
    ):
        # If the entire message is inside an unclosed think tag, return empty string
        text = ""

    return text.strip()
