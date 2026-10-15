"""Unit tests for RelayAuipBridge."""

from __future__ import annotations

import asyncio

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
)
from genai_tk.protocols.auip.relay_bridge import RelayAuipBridge


def test_bridge_translates_tool_start_and_end() -> None:
    events: list[AuipEvent] = []
    bridge = RelayAuipBridge(session_id="s123", on_event=events.append)

    # Tool start scope
    start_ev = {
        "kind": "scope",
        "scope_category": "start",
        "category": "tool",
        "name": "sql_query",
        "uuid": "tool-1",
        "timestamp": "2026-10-15T20:00:00Z",
        "category_profile": {
            "tool_request": {"args": {"query": "SELECT count(*) FROM users"}},
        },
    }
    bridge.handle_relay_event(start_ev)

    assert len(events) == 1
    call_ev = events[0]
    assert isinstance(call_ev, AuipToolCallEvent)
    assert call_ev.tool_name == "sql_query"
    assert call_ev.call_id == "tool-1"
    assert call_ev.args == {"query": "SELECT count(*) FROM users"}
    assert call_ev.session_id == "s123"

    # Tool end scope
    end_ev = {
        "kind": "scope",
        "scope_category": "end",
        "category": "tool",
        "name": "sql_query",
        "uuid": "tool-1",
        "timestamp": "2026-10-15T20:00:01Z",
        "category_profile": {
            "tool_response": {"output": "42"},
        },
    }
    bridge.handle_relay_event(end_ev)

    assert len(events) == 2
    res_ev = events[1]
    assert isinstance(res_ev, AuipToolResultEvent)
    assert res_ev.tool_name == "sql_query"
    assert res_ev.content == "42"
    assert res_ev.duration_ms == 1000.0
    assert res_ev.status == "success"


def test_bridge_translates_marks() -> None:
    events: list[AuipEvent] = []
    bridge = RelayAuipBridge(on_event=events.append)

    # skill.load mark
    skill_ev = {
        "kind": "mark",
        "name": "skill.load",
        "data": {"skill_name": "docgraph", "path": "skills/docgraph/SKILL.md"},
    }
    bridge.handle_relay_event(skill_ev)
    assert len(events) == 1
    assert isinstance(events[0], AuipArtifactEvent)
    assert events[0].artifact_type == "skill"
    assert "docgraph" in events[0].title

    # plan.update mark
    plan_ev = {
        "kind": "mark",
        "name": "plan.update",
        "data": {"todos": [{"content": "step 1", "status": "completed"}]},
    }
    bridge.handle_relay_event(plan_ev)
    assert len(events) == 2
    assert isinstance(events[1], AuipPhaseEvent)
    assert events[1].phase == "planner"
    assert len(events[1].state["todos"]) == 1

    # hitl.request mark
    hitl_ev = {
        "kind": "mark",
        "name": "hitl.request",
        "data": {"question": "Confirm deletion?", "type": "confirmation"},
    }
    bridge.handle_relay_event(hitl_ev)
    assert len(events) == 3
    assert isinstance(events[2], AuipHitlRequestEvent)
    assert events[2].question == "Confirm deletion?"


def test_bridge_translates_llm_and_root_agent_end() -> None:
    events: list[AuipEvent] = []
    bridge = RelayAuipBridge(on_event=events.append)

    llm_ev = {
        "kind": "scope",
        "scope_category": "end",
        "category": "llm",
        "uuid": "llm-1",
        "category_profile": {
            "model": "gpt-4o",
            "annotated_response": {
                "usage": {"prompt_tokens": 120, "completion_tokens": 45},
            },
        },
    }
    bridge.handle_relay_event(llm_ev)
    assert len(events) == 1
    assert isinstance(events[0], AuipUsageTelemetryEvent)
    assert events[0].input_tokens == 120
    assert events[0].output_tokens == 45

    root_end = {
        "kind": "scope",
        "scope_category": "end",
        "category": "agent",
        "uuid": "root-1",
        "parent_uuid": None,
    }
    bridge.handle_relay_event(root_end)
    assert len(events) == 2
    assert isinstance(events[1], AuipEndEvent)


def test_bridge_queue_mode() -> None:
    queue: asyncio.Queue[AuipEvent] = asyncio.Queue()
    bridge = RelayAuipBridge(queue=queue)

    bridge.handle_relay_event(
        {
            "kind": "mark",
            "name": "skill.load",
            "data": {"skill_name": "testing"},
        }
    )

    assert not queue.empty()
    item = queue.get_nowait()
    assert isinstance(item, AuipArtifactEvent)
