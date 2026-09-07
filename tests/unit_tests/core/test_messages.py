"""Unit tests for generic content block extraction and message decomposition."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from genai_tk.core.messages import (
    extract_ai_message_parts,
    extract_content_blocks,
    extract_text_content,
    extract_thinking_content,
    strip_reasoning_tags,
)


@pytest.mark.unit
class TestExtractContentBlocks:
    """Test standard and non-standard content block extraction."""

    def test_plain_string_message(self) -> None:
        msg = AIMessage(content="Hello world")
        blocks = extract_content_blocks(msg)
        assert blocks == [{"type": "text", "text": "Hello world"}]

    def test_structured_content_list(self) -> None:
        msg = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Analyzing requirements..."},
                {"type": "text", "text": "Here is the summary."},
            ]
        )
        blocks = extract_content_blocks(msg)
        assert len(blocks) == 2
        assert blocks[0] == {"type": "reasoning", "reasoning": "Analyzing requirements..."}
        assert blocks[1] == {"type": "text", "text": "Here is the summary."}

    def test_reasoning_in_additional_kwargs(self) -> None:
        msg = AIMessage(
            content="Final answer: 42",
            additional_kwargs={"reasoning_content": "20 + 22 = 42"},
        )
        blocks = extract_content_blocks(msg)
        assert len(blocks) == 2
        assert blocks[0] == {"type": "reasoning", "reasoning": "20 + 22 = 42"}
        assert blocks[1] == {"type": "text", "text": "Final answer: 42"}

    def test_thought_in_additional_kwargs(self) -> None:
        msg = AIMessage(
            content="Result",
            additional_kwargs={"thought": "internal chain of thought"},
        )
        blocks = extract_content_blocks(msg)
        assert len(blocks) == 2
        assert blocks[0] == {"type": "reasoning", "reasoning": "internal chain of thought"}
        assert blocks[1] == {"type": "text", "text": "Result"}

    def test_tool_calls_in_blocks(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "GDP"}, "id": "call_123"}],
        )
        blocks = extract_content_blocks(msg)
        assert any(b.get("type") == "tool_call" and b.get("name") == "search" for b in blocks)

    def test_wrapped_model_response(self) -> None:
        inner = AIMessage(content="Inner response text")
        wrapper = type("ModelResponse", (), {"result": [inner]})()
        blocks = extract_content_blocks(wrapper)
        assert blocks == [{"type": "text", "text": "Inner response text"}]


@pytest.mark.unit
class TestExtractAIMessageParts:
    """Test decomposing messages into text, thinking, and tool calls."""

    def test_clean_text_and_thinking_separation(self) -> None:
        msg = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Let me compute 100 * 5.\n"},
                {"type": "text", "text": "The answer is 500."},
            ]
        )
        parts = extract_ai_message_parts(msg)
        assert parts.text == "The answer is 500."
        assert parts.thinking == "Let me compute 100 * 5.\n"
        assert parts.tool_calls == []

    def test_extract_text_and_thinking_helpers(self) -> None:
        msg = AIMessage(
            content="Visible text only",
            additional_kwargs={"reasoning_content": "Hidden thinking trace"},
        )
        assert extract_text_content(msg) == "Visible text only"
        assert extract_thinking_content(msg) == "Hidden thinking trace"

    def test_streaming_chunks(self) -> None:
        chunk_think = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "thinking chunk"})
        assert extract_thinking_content(chunk_think) == "thinking chunk"
        assert extract_text_content(chunk_think) == ""

        chunk_text = AIMessageChunk(content="text chunk")
        assert extract_thinking_content(chunk_text) == ""
        assert extract_text_content(chunk_text) == "text chunk"


@pytest.mark.unit
class TestStripReasoningTags:
    """Test sanitizing leaked markdown/model reasoning tags."""

    def test_strip_closed_think_tags(self) -> None:
        raw = "<think>Step 1: check table.\nStep 2: sum values.</think>\n\nThe total is $45,000."
        assert strip_reasoning_tags(raw) == "The total is $45,000."

    def test_strip_thought_and_reasoning_tags(self) -> None:
        raw1 = "<thought>Thinking...</thought>Output 1"
        assert strip_reasoning_tags(raw1) == "Output 1"

        raw2 = "<reasoning>Analyzing...</reasoning>Output 2"
        assert strip_reasoning_tags(raw2) == "Output 2"

    def test_strip_assistant_final_marker(self) -> None:
        raw = "assistantanalysis Let's check table 5. assistantfinal 92,000,000"
        assert strip_reasoning_tags(raw) == "92,000,000"

    def test_strip_truncated_unclosed_think_tag(self) -> None:
        raw = "<think>The agent was interrupted while thinking"
        assert strip_reasoning_tags(raw) == ""
