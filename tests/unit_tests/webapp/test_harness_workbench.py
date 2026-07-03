"""Unit tests for the shared harness workbench (pure logic, no Streamlit rendering)."""

from collections.abc import AsyncIterator

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.events import (
    EndEvent,
    NodeEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from genai_tk.webapp.ui_components.harness_workbench import (
    extract_final_artifacts,
    extract_tool_artifact,
    lang_from_path,
    stream_harness_turn,
)


class _FakeHarness(BaseHarness):
    """Emits a canned event sequence, exactly matching a real harness's shape."""

    name = "fake"

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


def test_lang_from_path_infers_known_extensions() -> None:
    assert lang_from_path("script.py") == "python"
    assert lang_from_path("notes.md") == "markdown"
    assert lang_from_path("unknown.xyz") == "text"


def test_extract_tool_artifact_python_code() -> None:
    artifact = extract_tool_artifact("python_repl", {"code": "print(1)"}, "1")
    assert artifact is not None
    assert artifact.type == "code"
    assert artifact.language == "python"


def test_extract_tool_artifact_returns_none_for_unknown_tool() -> None:
    assert extract_tool_artifact("unknown_tool", {}, "result") is None


def test_extract_final_artifacts_detects_mermaid() -> None:
    text = "Here is a diagram:\n```mermaid\ngraph TD; A-->B;\n```"
    artifacts = extract_final_artifacts(text)
    assert len(artifacts) == 1
    assert artifacts[0].type == "mermaid"


def test_extract_final_artifacts_detects_code_block() -> None:
    text = "```python\nprint('hi')\n```"
    artifacts = extract_final_artifacts(text)
    assert len(artifacts) == 1
    assert artifacts[0].type == "code"
    assert artifacts[0].language == "python"


def test_stream_harness_turn_builds_trace_steps_and_text() -> None:
    events: list[StreamEvent] = [
        NodeEvent(node="researcher", state={}),
        ToolCallEvent(tool_name="web_search_tool", args={"query": "AI"}, call_id="1"),
        ToolResultEvent(tool_name="web_search_tool", content="some results", call_id="1"),
        TokenEvent(text="The answer is 42."),
        EndEvent(),
    ]
    harness = _FakeHarness(events)

    result = stream_harness_turn(harness, "What is the answer?")

    assert result.text == "The answer is 42."
    assert not result.is_clarification
    node_names = [s.node for s in result.steps]
    assert "researcher" in node_names
    researcher_step = next(s for s in result.steps if s.node == "researcher")
    assert researcher_step.tools[0].name == "web_search_tool"
    assert researcher_step.tools[0].result == "some results"


def test_stream_harness_turn_handles_clarification() -> None:
    events: list[StreamEvent] = [
        NodeEvent(node="planner", state={}),
    ]

    from genai_tk.agents.harness.events import ClarificationEvent

    events.append(ClarificationEvent(question="Which city?"))
    harness = _FakeHarness(events)

    result = stream_harness_turn(harness, "Tell me the weather")

    assert result.is_clarification
    assert result.text == "Which city?"
