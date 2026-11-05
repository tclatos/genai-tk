"""Unit tests for the trajectory store write side (run grouping + index upsert).

One run used to be split across several directories: a scope whose ``parent_uuid``
had not been delivered yet — or never is, which is the case for NeMo Relay's
decision-model evaluator scopes, emitted from worker threads with a parent id that
never appears as an event uuid — was mistaken for a new root, and every root change
reset the run's aggregate. An Alko classification therefore ended up with two
``index.jsonl`` lines reporting 5 and 9 of its 14 LLM calls.

These tests pin the two properties that make ``index.jsonl``, ``meta.json`` and
``events.jsonl`` agree: an unknown parent never invents a run, and a run is
finalized with one index line no matter how often it is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genai_tk.extra.monitoring.nemo_relay_setup import _StoreState

pytestmark = pytest.mark.unit

# The LangGraph trace root: the top-level agent scope's parent, never instrumented.
PHANTOM_ROOT = "01a1a88f-9963-7330-8b5e-9852fc66b4ae"
AGENT = "01a1a88f-9963-7330-8b5e-9876f77f1358"
LLM_1 = "01a1a88f-a001-7000-8000-000000000001"
LLM_2 = "01a1a88f-a002-7000-8000-000000000002"
TOOL = "01a1a890-4bf6-7fa0-8e33-2d3190d66134"
# The tool scope the evaluator events *claim* as parent. NeMo Relay records the
# native handle of the thread-local top of stack, which is not always the uuid of
# an emitted scope event — so this id never shows up as an event uuid.
ORPHAN_PARENT = "01a1a890-4bf6-7fa0-8e33-2d6d8a6a1325"
EVALUATOR = "01a1a890-4bf7-7202-b8ba-047e712c67ad"
VERDICT = "01a1a890-5254-7503-ae2d-b23204bd0957"


def _scope(
    uuid: str,
    parent: str | None,
    cat: str,
    sc: str,
    name: str,
    ts: str,
    *,
    usage: tuple[int, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ATOF scope event, optionally carrying LLM usage."""
    profile: dict[str, Any] | None = None
    if usage is not None:
        profile = {
            "annotated_response": {
                "model": name,
                "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
            }
        }
    return {
        "kind": "scope",
        "scope_category": sc,
        "atof_version": "0.1",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": ts,
        "name": name,
        "category": cat,
        "category_profile": profile,
        "data": None,
        "metadata": metadata,
    }


def _mark(uuid: str, parent: str, name: str, ts: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one ATOF mark event."""
    return {
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": ts,
        "name": name,
        "data": data,
        "metadata": None,
    }


def _feed(store: _StoreState, events: list[dict[str, Any]]) -> None:
    """Write events through the store exactly as the Relay subscriber does."""
    for event in events:
        store.write(event, json.dumps(event))


def _index_lines(store_dir: Path) -> list[dict[str, Any]]:
    """Return the parsed ``index.jsonl`` lines."""
    path = store_dir / "index.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_dirs(store_dir: Path) -> list[str]:
    """Return the run directory names in the store."""
    return sorted(p.name for p in store_dir.iterdir() if p.is_dir())


def test_orphan_parent_does_not_split_the_run(tmp_path: Path) -> None:
    """Evaluator events whose parent id is never an event uuid stay in the run."""
    store = _StoreState(tmp_path)
    _feed(
        store,
        [
            _scope(AGENT, PHANTOM_ROOT, "agent", "start", "Resource Unit Classifier", "2026-11-04T21:55:45Z"),
            _scope(LLM_1, AGENT, "llm", "start", "deepseek", "2026-11-04T21:55:46Z"),
            _scope(LLM_1, AGENT, "llm", "end", "deepseek", "2026-11-04T21:55:47Z", usage=(1000, 100)),
            _scope(TOOL, AGENT, "tool", "start", "navigate_folder", "2026-11-04T21:55:48Z"),
            # The decision-model evaluator scope: its parent is not an event uuid.
            _scope(EVALUATOR, ORPHAN_PARENT, "evaluator", "start", "decision.clef-flash", "2026-11-04T21:55:49Z"),
            _mark(VERDICT, EVALUATOR, "decision.verdict", "2026-11-04T21:55:50Z"),
            _scope(EVALUATOR, ORPHAN_PARENT, "evaluator", "end", "decision.clef-flash", "2026-11-04T21:55:50Z"),
            _scope(TOOL, AGENT, "tool", "end", "navigate_folder", "2026-11-04T21:55:51Z"),
            _scope(LLM_2, AGENT, "llm", "start", "deepseek", "2026-11-04T21:55:52Z"),
            _scope(LLM_2, AGENT, "llm", "end", "deepseek", "2026-11-04T21:55:53Z", usage=(2000, 200)),
        ],
    )
    store.close()

    assert _run_dirs(tmp_path) == [PHANTOM_ROOT], "an unknown parent must not create a run of its own"

    lines = _index_lines(tmp_path)
    assert len(lines) == 1, f"expected one index line for one run, got {len(lines)}"
    line = lines[0]
    assert line["n_llm_calls"] == 2
    assert line["n_tool_calls"] == 1
    assert line["total_prompt_tokens"] == 3000
    assert line["total_completion_tokens"] == 300
    assert line["ended_at"] == "2026-11-04T21:55:53Z"

    # Every event stayed in the run's own file: index, meta and events agree.
    events = [
        json.loads(ln) for ln in (tmp_path / PHANTOM_ROOT / "events.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(events) == 10
    meta = json.loads((tmp_path / PHANTOM_ROOT / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_llm_calls"] == line["n_llm_calls"]
    assert meta["n_tool_calls"] == line["n_tool_calls"]
    assert meta["total_prompt_tokens"] == line["total_prompt_tokens"]
    assert meta["profile"] == "Resource Unit Classifier"


def test_late_parent_event_is_folded_into_the_open_run(tmp_path: Path) -> None:
    """A child arriving before its parent's own event joins the run, not a new one."""
    store = _StoreState(tmp_path)
    _feed(
        store,
        [
            _scope(AGENT, PHANTOM_ROOT, "agent", "start", "Resource Unit Classifier", "2026-11-04T21:55:45Z"),
            # Child first, parent second: the parent's own start event is late.
            _scope(EVALUATOR, ORPHAN_PARENT, "evaluator", "start", "decision.clef-flash", "2026-11-04T21:55:46Z"),
            _scope(ORPHAN_PARENT, AGENT, "tool", "start", "navigate_folder", "2026-11-04T21:55:47Z"),
            _scope(ORPHAN_PARENT, AGENT, "tool", "end", "navigate_folder", "2026-11-04T21:55:48Z"),
        ],
    )
    store.close()

    assert _run_dirs(tmp_path) == [PHANTOM_ROOT]
    assert len(_index_lines(tmp_path)) == 1
    assert _index_lines(tmp_path)[0]["n_tool_calls"] == 1


def test_two_agent_runs_are_separate_and_each_has_one_index_line(tmp_path: Path) -> None:
    """A second top-level agent scope starts its own run even while one is open."""
    second_root = "01a1a999-0000-7000-8000-000000000001"
    second_agent = "01a1a999-0000-7000-8000-000000000002"
    store = _StoreState(tmp_path)
    _feed(
        store,
        [
            _scope(AGENT, PHANTOM_ROOT, "agent", "start", "Resource Unit Classifier", "2026-11-04T21:55:45Z"),
            _scope(LLM_1, AGENT, "llm", "start", "deepseek", "2026-11-04T21:55:46Z"),
            _scope(second_agent, second_root, "agent", "start", "RFQ Extractor", "2026-11-04T21:55:47Z"),
            _scope(LLM_2, second_agent, "llm", "start", "deepseek", "2026-11-04T21:55:48Z"),
        ],
    )
    store.close()

    assert _run_dirs(tmp_path) == sorted([PHANTOM_ROOT, second_root])
    lines = {line["run_id"]: line for line in _index_lines(tmp_path)}
    assert set(lines) == {PHANTOM_ROOT, second_root}
    assert lines[PHANTOM_ROOT]["n_llm_calls"] == 1
    assert lines[second_root]["n_llm_calls"] == 1
    assert lines[second_root]["profile"] == "RFQ Extractor"


def test_finalizing_twice_keeps_one_index_line_with_the_final_totals(tmp_path: Path) -> None:
    """Re-finalizing rewrites the run's index line instead of appending a partial one."""
    store = _StoreState(tmp_path)
    _feed(
        store,
        [
            _scope(AGENT, PHANTOM_ROOT, "agent", "start", "Resource Unit Classifier", "2026-11-04T21:55:45Z"),
            _scope(LLM_1, AGENT, "llm", "start", "deepseek", "2026-11-04T21:55:46Z"),
            _scope(LLM_1, AGENT, "llm", "end", "deepseek", "2026-11-04T21:55:47Z", usage=(500, 50)),
        ],
    )
    store.close()  # first finalize: 1 LLM call

    store.write(_scope(LLM_2, AGENT, "llm", "start", "deepseek", "2026-11-04T21:55:52Z"), "{}")
    store.close()  # second finalize: same run, now 2 LLM calls

    lines = _index_lines(tmp_path)
    assert len(lines) == 1, "a run must never own two index lines"
    assert lines[0]["n_llm_calls"] == 2, "the run's aggregate must not reset between finalizes"
    assert lines[0]["total_prompt_tokens"] == 500
    assert lines[0]["ended_at"] == "2026-11-04T21:55:52Z"


def test_index_read_dedupes_legacy_duplicate_lines(tmp_path: Path) -> None:
    """The reader keeps the last line for a run id, so legacy stores report the final totals."""
    from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore

    run_dir = tmp_path / PHANTOM_ROOT
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"run_id": PHANTOM_ROOT, "profile": "Resource Unit Classifier", "n_llm_calls": 5})
        + "\n"
        + json.dumps({"run_id": PHANTOM_ROOT, "profile": "navigate_folder", "n_llm_calls": 9})
        + "\n",
        encoding="utf-8",
    )

    summaries = TrajectoryStore(tmp_path).list_runs()
    assert len(summaries) == 1
    assert summaries[0].n_llm_calls == 9
