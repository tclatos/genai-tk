"""Export the trajectory store to harbor's ATIF web viewer layout.

``cli trajectory view`` serves the store through the external ``harbor``
viewer (``harbor view --jobs``). Harbor expects a jobs tree — one directory
per job, each holding trial directories with ``config.json``, ``result.json``
and an ATIF ``agent/trajectory.json`` — while the trajectory store keeps one
directory per run with raw ATOF ``events.jsonl``. This module projects each
recorded ATOF run into a valid harbor job (one job per run, one trial) so the
viewer can render run content.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from genai_tk.extra.monitoring.trajectory_store import (
    Trajectory,
    TrajectoryStore,
    short_model_name,
)

_TRIAL_NAME = "run"

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def export_store_to_harbor(store: TrajectoryStore, out_dir: Path) -> Path:
    """Export every recorded run as a harbor job; returns the export root.

    Args:
        store: Trajectory store to read runs from.
        out_dir: Export root (regenerated on every call).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = 0
    for run in store.list_runs():
        traj = store.get(run.run_id)
        if traj is None:
            continue
        _export_run(store, traj, out_dir)
        n_jobs += 1
    logger.debug(f"harbor_export: exported {n_jobs} run(s) → {out_dir}")
    return out_dir


def _export_run(store: TrajectoryStore, traj: Trajectory, out_dir: Path) -> None:
    """Write one run as a harbor job directory with a single trial."""
    trial_config: dict[str, Any] = {
        "task": {"path": str(store.root / traj.run_id)},
        "trial_name": _TRIAL_NAME,
        "trials_dir": str(out_dir),
        "agent": {"name": traj.profile or "agent"},
    }
    trial_result: dict[str, Any] = {
        "task_name": _slug(traj.profile) or "trajectory",
        "trial_name": _TRIAL_NAME,
        "trial_uri": str(out_dir / _job_dir_name(traj) / _TRIAL_NAME),
        "task_id": {"path": str(store.root / traj.run_id)},
        "task_checksum": "atof",
        "config": trial_config,
        "agent_info": _agent_info(traj),
        "started_at": _iso_or_none(traj.started_at),
        "finished_at": _iso_or_none(traj.ended_at),
    }

    trial_dir = out_dir / _job_dir_name(traj) / _TRIAL_NAME
    (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps(trial_config, indent=2), encoding="utf-8")
    (trial_dir / "result.json").write_text(json.dumps(trial_result, indent=2), encoding="utf-8")
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_to_atif(store, traj), indent=2), encoding="utf-8")


def _agent_info(traj: Trajectory) -> dict[str, Any]:
    model = traj.llm_calls[0].model if traj.llm_calls else None
    info: dict[str, Any] = {"name": traj.profile or "agent", "version": "atof-0.1"}
    if model:
        info["model_info"] = {"name": model}
    return info


def _job_dir_name(traj: Trajectory) -> str:
    """Filesystem-safe, reverse-sortable job name: <ts>_<profile>_<run8>."""
    ts = ""
    started = _parse_iso(traj.started_at)
    if started is not None:
        ts = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [p for p in (ts, _slug(traj.profile), traj.run_id[:8]) if p]
    return "-".join(parts) or traj.run_id


def _to_atif(store: TrajectoryStore, traj: Trajectory) -> dict[str, Any]:
    """Project a Trajectory to a harbor-valid ATIF v1.7 document."""
    steps: list[dict[str, Any]] = []

    user_msg = store._root_user_message(traj)
    if user_msg:
        steps.append({"step_id": len(steps) + 1, "source": "user", "message": user_msg})

    for turn in traj.turns:
        lc = turn.llm_call
        if lc is None:
            # Standalone tools/skills without an LLM call.
            steps.extend(_tool_steps(turn.tool_calls, step_id_base=len(steps) + 1))
            continue

        requested_ids = [str(req.get("id")) for req in lc.tool_calls if isinstance(req, dict) and req.get("id")]
        tool_calls = [
            {
                "tool_call_id": str(req.get("id") or f"call-{i + 1}"),
                "function_name": str(req.get("name") or "unknown"),
                "arguments": req.get("arguments") if isinstance(req.get("arguments"), dict) else {},
            }
            for i, req in enumerate(lc.tool_calls)
            if isinstance(req, dict)
        ]
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": lc.message or "",
            "tool_calls": tool_calls or None,
        }
        if lc.model:
            step["model_name"] = lc.model
        usage = lc.usage or {}
        if usage:
            step["metrics"] = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        results = _observation_results(turn.tool_calls, set(requested_ids))
        if results:
            step["observation"] = {"results": results}
        steps.append(step)

    if not steps:
        steps.append({"step_id": 1, "source": "user", "message": user_msg or "(no events recorded)"})

    return {
        "schema_version": "ATIF-v1.7",
        "session_id": traj.run_id,
        "agent": {
            "name": traj.profile or "agent",
            "version": "atof-0.1",
            "model_name": short_model_name(traj.llm_calls[0].model) if traj.llm_calls else None,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": traj.total_prompt_tokens,
            "total_completion_tokens": traj.total_completion_tokens,
            "total_steps": len(steps),
        },
    }


def _tool_steps(tool_calls: list[Any], *, step_id_base: int) -> list[dict[str, Any]]:
    """Render tools without a parent LLM call as deterministic agent steps."""
    steps: list[dict[str, Any]] = []
    for offset, tc in enumerate(tool_calls):
        steps.append(
            {
                "step_id": step_id_base + offset,
                "source": "agent",
                "message": "",
                "observation": {"results": [{"content": tc.result}]},
            }
        )
    return steps


def _observation_results(tool_calls: list[Any], valid_ids: set[str]) -> list[dict[str, Any]]:
    """Map executed tool results to ATIF observation results.

    ``source_call_id`` is only set when the executed tool matches a tool call
    requested in the same step (harbor validates references).
    """
    results: list[dict[str, Any]] = []
    for tc in tool_calls:
        entry: dict[str, Any] = {"content": tc.result}
        if tc.tool_call_id and tc.tool_call_id in valid_ids:
            entry["source_call_id"] = tc.tool_call_id
        results.append(entry)
    return results


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip()).strip("-")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iso_or_none(ts: str | None) -> str | None:
    return ts if _parse_iso(ts) is not None else None
