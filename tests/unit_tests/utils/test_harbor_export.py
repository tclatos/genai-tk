"""Unit tests for the harbor ATIF viewer export (no harbor install required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genai_tk.extra.monitoring.harbor_export import export_store_to_harbor
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore

pytestmark = pytest.mark.unit

_ROOT = "01ROOT0000-0000-0000-0000-000000000001"
_LLM1 = "01LLM10000-0000-0000-0000-000000000001"
_TOOL1 = "01TOOL0000-0000-0000-0000-000000000001"


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
    return {
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


def _sample_events() -> list[dict[str, Any]]:
    return [
        _scope(
            _ROOT,
            _ROOT,
            "agent",
            "start",
            "test-profile",
            ts="2026-08-21T12:00:00Z",
            data={"messages": "Please echo 'hello'."},
        ),
        _scope(_LLM1, _ROOT, "llm", "start", "gpt-oss-120b", ts="2026-08-21T12:00:01Z"),
        _scope(
            _LLM1,
            _ROOT,
            "llm",
            "end",
            "gpt-oss-120b",
            ts="2026-08-21T12:00:02Z",
            category_profile={
                "annotated_response": {
                    "model": "gpt-oss-120b",
                    "message": "",
                    "tool_calls": [{"name": "echo", "arguments": {"message": "hello"}, "id": "tc1"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            },
        ),
        _scope(_TOOL1, _ROOT, "tool", "start", "echo", ts="2026-08-21T12:00:02Z", data={"message": "hello"}),
        _scope(
            _TOOL1,
            _ROOT,
            "tool",
            "end",
            "echo",
            ts="2026-08-21T12:00:03Z",
            data={"data": {"content": "echo:hello", "tool_call_id": "tc1"}},
        ),
        _scope(_ROOT, _ROOT, "agent", "end", "test-profile", ts="2026-08-21T12:00:05Z"),
    ]


def _populated_store(tmp_path: Path) -> TrajectoryStore:
    run_dir = tmp_path / "store" / _ROOT
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in _sample_events()) + "\n", encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": _ROOT,
                "profile": "test-profile",
                "started_at": "2026-08-21T12:00:00Z",
                "ended_at": "2026-08-21T12:00:05Z",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "store" / "index.jsonl").write_text(
        json.dumps({"run_id": _ROOT, "profile": "test-profile", "started_at": "2026-08-21T12:00:00Z"}),
        encoding="utf-8",
    )
    return TrajectoryStore(root=tmp_path / "store")


def test_export_creates_harbor_jobs_layout(tmp_path: Path) -> None:
    store = _populated_store(tmp_path)
    out_dir = tmp_path / "export"

    export_store_to_harbor(store, out_dir)

    job_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(job_dirs) == 1
    job = job_dirs[0]
    assert "test-profile" in job.name
    assert "20260821" in job.name

    trial = job / "run"
    assert (trial / "config.json").exists()
    assert (trial / "result.json").exists()
    assert (trial / "agent" / "trajectory.json").exists()


def test_export_trajectory_is_valid_atif(tmp_path: Path) -> None:
    store = _populated_store(tmp_path)
    out_dir = tmp_path / "export"

    export_store_to_harbor(store, out_dir)

    job = next(d for d in out_dir.iterdir() if d.is_dir())
    atif = json.loads((job / "run" / "agent" / "trajectory.json").read_text(encoding="utf-8"))

    assert atif["schema_version"] == "ATIF-v1.7"
    assert atif["session_id"] == _ROOT
    # Sequential step ids starting from 1 (harbor validates this).
    assert [s["step_id"] for s in atif["steps"]] == [1, 2]
    user_step, agent_step = atif["steps"]
    assert user_step["source"] == "user"
    assert "echo" in user_step["message"]
    assert agent_step["source"] == "agent"
    assert agent_step["tool_calls"][0]["function_name"] == "echo"
    assert agent_step["observation"]["results"][0]["source_call_id"] == "tc1"
    assert agent_step["metrics"]["prompt_tokens"] == 100


def test_export_result_json_parses(tmp_path: Path) -> None:
    store = _populated_store(tmp_path)
    out_dir = tmp_path / "export"

    export_store_to_harbor(store, out_dir)

    job = next(d for d in out_dir.iterdir() if d.is_dir())
    result = json.loads((job / "run" / "result.json").read_text(encoding="utf-8"))
    assert result["trial_name"] == "run"
    assert result["agent_info"]["name"] == "test-profile"
    assert result["config"]["task"]["path"].endswith(_ROOT)


def test_export_empty_store_creates_empty_dir(tmp_path: Path) -> None:
    store_dir = tmp_path / "empty_store"
    store_dir.mkdir()
    out_dir = tmp_path / "export"

    export_store_to_harbor(TrajectoryStore(root=store_dir), out_dir)

    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []
