"""Unit tests for RichAuipRenderer."""

from __future__ import annotations

import io

from rich.console import Console

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipErrorEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
)
from genai_tk.protocols.auip.renderers.rich_console import RichAuipRenderer


def _buffer_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    return console, buf


def test_renderer_tokens_and_finalize() -> None:
    console, buf = _buffer_console()
    renderer = RichAuipRenderer(console=console, raw=True)

    renderer.render_event(AuipTokenEvent(text="Hello "))
    renderer.render_event(AuipTokenEvent(text="world!"))
    out = renderer.finalize()

    assert out == "Hello world!"
    assert "Hello world!" in buf.getvalue()


def test_renderer_tool_call_and_result() -> None:
    console, buf = _buffer_console()
    renderer = RichAuipRenderer(console=console)

    renderer.render_event(AuipToolCallEvent(tool_name="echo", args={"msg": "hi"}))
    renderer.render_event(AuipToolResultEvent(tool_name="echo", content="hi", duration_ms=25.0))
    renderer.render_event(AuipEndEvent())

    output = buf.getvalue()
    assert "[tool]" in output
    assert "echo" in output
    assert "[tool result]" in output
    assert "25ms" in output


def test_renderer_planning_and_skills() -> None:
    console, buf = _buffer_console()
    renderer = RichAuipRenderer(console=console)

    renderer.render_event(
        AuipPhaseEvent(
            phase="planner",
            state={"todos": [{"content": "Inspect code", "status": "in_progress"}]},
        )
    )
    renderer.render_event(
        AuipArtifactEvent(
            artifact_type="skill",
            title="docgraph",
            content="",
        )
    )

    output = buf.getvalue()
    assert "Planning" in output
    assert "Inspect code" in output
    assert "Loaded skill" in output


def test_renderer_hitl_and_error() -> None:
    console, buf = _buffer_console()
    renderer = RichAuipRenderer(console=console)

    renderer.render_event(AuipHitlRequestEvent(question="Proceed?"))
    renderer.render_event(AuipErrorEvent(message="Something failed"))

    output = buf.getvalue()
    assert "Proceed?" in output
    assert "Something failed" in output
