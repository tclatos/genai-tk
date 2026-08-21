"""Unit tests for the trajectory store read layer + Phase 3 store I/O (no API key).

Synthesizes a realistic ATOF event stream (matching the real NeMo Relay shape:
self-parented root agent scope, flat tool args, nested ToolMessage results) into
a tmp store and verifies:

- ``TrajectoryStore.list_runs`` / ``get`` / ``messages`` / ``skills`` / ``stats`` / ``diff``.
- ``load_trajectory_messages`` (Phase 3 eval I/O).
- ``compare_trajectory_to_golden`` (structural golden comparison).
- ``judge_trajectory`` deterministic judges (tool_use / grounding / efficiency).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genai_tk.agents.langchain.trajectory_store_io import (
    compare_trajectory_to_golden,
    judge_trajectory,
    load_trajectory_messages,
)
from genai_tk.utils.trajectory_store import TrajectoryStore

pytestmark = pytest.mark.unit

_ROOT = "01ROOT0000-0000-0000-0000-000000000001"
_LLM1 = "01LLM10000-0000-0000-0000-000000000001"
_TOOL1 = "01TOOL0000-0000-0000-0000-000000000001"
_LLM2 = "01LLM20000-0000-0000-0000-000000000002"


def _scope(
    uuid: str,
    parent: str,
    cat: str,
    sc: str,
    name: str,
    *,
    ts: str,
    data: dict[str, Any] | None = None,
    category_profile: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    e: dict[str, Any] = {
        "kind": "scope",
        "scope_category": sc,
        "atof_version": "0.1",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": ts,
        "name": name,
        "attributes": [],
        "category": cat,
        "category_profile": category_profile,
        "data": data,
        "data_schema": None,
        "metadata": metadata,
    }
    return e


def _mark(uuid: str, parent: str, name: str, ts: str, data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": ts,
        "name": name,
        "data": data,
        "data_schema": None,
        "metadata": metadata,
    }


def _write_run(store_dir: Path, run_id: str, events: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """Write a run's events.jsonl + meta.json + index.jsonl into a tmp store."""
    run_dir = store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    idx = store_dir / "index.jsonl"
    line = {
        "run_id": run_id,
        "profile": meta["profile"],
        "started_at": meta["started_at"],
        "ended_at": meta.get("ended_at"),
        "status": meta.get("status", "ok"),
        "n_llm_calls": meta.get("n_llm_calls", 0),
        "n_tool_calls": meta.get("n_tool_calls", 0),
        "total_prompt_tokens": meta.get("total_prompt_tokens", 0),
        "total_completion_tokens": meta.get("total_completion_tokens", 0),
        "tools": meta.get("tools", []),
        "skills_loaded": meta.get("skills_loaded", []),
    }
    with idx.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def _sample_events() -> list[dict[str, Any]]:
    """A realistic deep-agent ATOF stream: user → echo tool → final answer."""
    return [
        # Root agent scope (self-parented), carries the user message in data.messages.
        _scope(
            _ROOT,
            _ROOT,
            "agent",
            "start",
            "test-profile",
            ts="2026-08-21T12:00:00Z",
            data={"messages": "Please echo 'hello' using the echo tool."},
        ),
        # First LLM call: emits an echo tool call.
        _scope(_LLM1, _ROOT, "llm", "start", "gpt-oss-120b", ts="2026-08-21T12:00:01Z"),
        _scope(
            _LLM1,
            _ROOT,
            "llm",
            "end",
            "gpt-oss-120b",
            ts="2026-08-21T12:00:02Z",
            category_profile={
                "model_name": "gpt-oss-120b",
                "annotated_response": {
                    "model": "gpt-oss-120b",
                    "message": "",
                    "tool_calls": [{"name": "echo", "arguments": {"message": "hello"}, "id": "tc1"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                },
            },
        ),
        # Echo tool call.
        _scope(_TOOL1, _ROOT, "tool", "start", "echo", ts="2026-08-21T12:00:02Z", data={"message": "hello"}),
        _scope(
            _TOOL1,
            _ROOT,
            "tool",
            "end",
            "echo",
            ts="2026-08-21T12:00:03Z",
            data={
                "__nv_pydantic__": "langchain_core.messages.tool.ToolMessage",
                "data": {"content": "echo:hello", "name": "echo", "tool_call_id": "tc1", "status": "success"},
            },
        ),
        # skill.load mark.
        _mark(
            "01MARK0000-0000-0000-0000-000000000001",
            _ROOT,
            "skill.load",
            "2026-08-21T12:00:02Z",
            {"skill_name": "navigation"},
            {"skill_load_source": "structured_read"},
        ),
        # Second LLM call: final answer.
        _scope(_LLM2, _ROOT, "llm", "start", "gpt-oss-120b", ts="2026-08-21T12:00:03Z"),
        _scope(
            _LLM2,
            _ROOT,
            "llm",
            "end",
            "gpt-oss-120b",
            ts="2026-08-21T12:00:04Z",
            category_profile={
                "model_name": "gpt-oss-120b",
                "annotated_response": {
                    "model": "gpt-oss-120b",
                    "message": "The echo result is echo:hello.",
                    "tool_calls": [],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
                },
            },
        ),
        # Root agent scope end.
        _scope(
            _ROOT, _ROOT, "agent", "end", "test-profile", ts="2026-08-21T12:00:05Z", metadata={"otel.status_code": "OK"}
        ),
    ]


def _sample_meta() -> dict[str, Any]:
    return {
        "run_id": _ROOT,
        "profile": "test-profile",
        "started_at": "2026-08-21T12:00:00Z",
        "ended_at": "2026-08-21T12:00:05Z",
        "status": "ok",
        "n_scopes": 7,
        "n_llm_calls": 2,
        "n_tool_calls": 1,
        "total_prompt_tokens": 220,
        "total_completion_tokens": 30,
        "tools": ["echo"],
        "skills_loaded": ["navigation"],
    }


@pytest.fixture
def store(tmp_path: Path) -> TrajectoryStore:
    _write_run(tmp_path, _ROOT, _sample_events(), _sample_meta())
    return TrajectoryStore(root=tmp_path)


def test_list_runs(store: TrajectoryStore) -> None:
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == _ROOT
    assert runs[0].profile == "test-profile"
    assert runs[0].n_llm_calls == 2
    assert runs[0].n_tool_calls == 1
    assert runs[0].tools == ["echo"]
    assert runs[0].skills_loaded == ["navigation"]


def test_get_trajectory_reconstructs_llm_and_tool_calls(store: TrajectoryStore) -> None:
    traj = store.get(_ROOT)
    assert traj is not None
    assert traj.profile == "test-profile"
    assert len(traj.llm_calls) == 2
    assert traj.llm_calls[0].model == "gpt-oss-120b"
    assert traj.llm_calls[0].tool_calls[0]["name"] == "echo"
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].name == "echo"
    assert traj.tool_calls[0].args == {"message": "hello"}
    assert traj.tool_calls[0].result == "echo:hello"
    assert traj.tool_calls[0].tool_call_id == "tc1"
    assert traj.total_prompt_tokens == 220
    assert traj.total_completion_tokens == 30
    assert traj.skill_names == ["navigation"]


def test_messages_projection(store: TrajectoryStore) -> None:
    msgs = store.messages(_ROOT)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert msgs[0]["content"].startswith("Please echo")
    # First assistant carries the tool call.
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "echo"
    # Tool result message.
    assert msgs[2]["content"] == "echo:hello"
    # Final assistant answer.
    assert msgs[3]["content"].startswith("The echo result")


def test_skills_and_stats(store: TrajectoryStore) -> None:
    loads = store.skills(_ROOT)
    assert len(loads) == 1
    assert loads[0].skill_name == "navigation"
    stats = store.stats()
    assert stats["n_runs"] == 1
    assert stats["tool_frequency"] == {"echo": 1}
    assert stats["skill_load_frequency"] == {"navigation": 1}


def test_diff(store: TrajectoryStore) -> None:
    # Diff the run against itself → no tools/skills only-in-a/b.
    d = store.diff(_ROOT, _ROOT)
    assert d["tools_only_in_a"] == []
    assert d["tools_only_in_b"] == []


def test_load_trajectory_messages(store: TrajectoryStore) -> None:
    msgs = load_trajectory_messages(_ROOT, store=store)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]


def test_compare_trajectory_to_golden_pass(store: TrajectoryStore) -> None:
    golden = {"profile": "test-profile", "tools": ["echo"], "skills": ["navigation"], "min_steps": 4, "max_steps": 4}
    verdict = compare_trajectory_to_golden(_ROOT, golden, store=store)
    assert verdict["pass"] is True
    assert verdict["checks"]["tools_superset"] is True
    assert verdict["checks"]["skills_superset"] is True


def test_compare_trajectory_to_golden_fail_missing_tool(store: TrajectoryStore) -> None:
    golden = {"tools": ["nonexistent_tool"]}
    verdict = compare_trajectory_to_golden(_ROOT, golden, store=store)
    assert verdict["pass"] is False
    assert verdict["checks"]["tools_superset"] is False


def test_judge_trajectory_deterministic(store: TrajectoryStore) -> None:
    verdicts = judge_trajectory(
        _ROOT,
        [
            {"kind": "tool_use", "tools": ["echo"]},
            {"kind": "grounding"},
            {"kind": "efficiency", "max_repeat": 3},
        ],
        store=store,
    )
    by_kind = {v["kind"]: v for v in verdicts}
    assert by_kind["tool_use"]["score"] is True
    assert by_kind["grounding"]["score"] is True
    assert by_kind["efficiency"]["score"] is True


def test_judge_trajectory_efficiency_fails_on_repeats(tmp_path: Path) -> None:
    """An agent calling echo 5 times fails efficiency with max_repeat=3."""
    events: list[dict[str, Any]] = [_scope(_ROOT, _ROOT, "agent", "start", "p", ts="t0", data={"messages": "q"})]
    for i in range(5):
        tu = f"01TOOL{i:04d}-0000-0000-0000-000000000000"
        events.append(_scope(tu, _ROOT, "tool", "start", "echo", ts=f"t{i}a", data={"message": "x"}))
        events.append(
            _scope(
                tu,
                _ROOT,
                "tool",
                "end",
                "echo",
                ts=f"t{i}b",
                data={"data": {"content": "echo:x", "tool_call_id": f"tc{i}"}},
            )
        )
    events.append(_scope(_ROOT, _ROOT, "agent", "end", "p", ts="t9", metadata={"otel.status_code": "OK"}))
    meta = {
        "run_id": _ROOT,
        "profile": "p",
        "started_at": "t0",
        "ended_at": "t9",
        "status": "ok",
        "tools": ["echo"],
        "n_tool_calls": 5,
    }
    _write_run(tmp_path, _ROOT, events, meta)
    store = TrajectoryStore(root=tmp_path)
    verdicts = judge_trajectory(_ROOT, [{"kind": "efficiency", "max_repeat": 3}], store=store)
    assert verdicts[0]["score"] is False
