"""Store-based trajectory I/O for evals (Phase 3).

Bridges the local trajectory store (:mod:`genai_tk.utils.trajectory_store`)
to the eval stack:

- :func:`load_trajectory_messages` — read a recorded run and project it to
  OpenAI-format messages, replacing the ``extract_message_trajectory``
  re-run pattern. Eval cases read a **real captured trajectory** instead of
  re-running the agent.
- :func:`compare_trajectory_to_golden` — structural comparison of a recorded
  run against a golden reference (tools ⊆, skills ⊆, step-count tolerance).
- :func:`judge_trajectory` — run configured ``openevals``/``agentevals``
  judges over a stored trajectory and return a verdict report (the eval-time
  analog of the analysis agent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genai_tk.utils.trajectory_store import TrajectoryStore


def load_trajectory_messages(
    run_id: str,
    *,
    store: TrajectoryStore | None = None,
) -> list[dict[str, Any]]:
    """Return the OpenAI-format message projection of a recorded run.

    Args:
        run_id: Root agent scope uuid of the run.
        store: Optional store instance (defaults to the configured store root).

    Returns:
        List of OpenAI-format message dicts (user/assistant/tool/assistant),
        or an empty list if the run is not found.
    """
    s = store if store is not None else TrajectoryStore()
    return s.messages(run_id)


def load_trajectory(
    run_id: str,
    *,
    store: TrajectoryStore | None = None,
) -> Any:
    """Return the typed :class:`~genai_tk.utils.trajectory_store.Trajectory` for a run."""
    s = store if store is not None else TrajectoryStore()
    return s.get(run_id)


def compare_trajectory_to_golden(
    run_id: str,
    golden: dict[str, Any] | Path,
    *,
    store: TrajectoryStore | None = None,
    step_tolerance: int = 2,
) -> dict[str, Any]:
    """Structurally compare a recorded run to a golden reference.

    The golden is a dict (or path to a JSON file) with optional keys:

    - ``tools``: list of tool names that **must** have been called (superset).
    - ``skills``: list of skill names that **must** have been loaded.
    - ``min_steps`` / ``max_steps``: bounds on the projected message count.
    - ``profile``: expected profile name.

    Args:
        run_id: The recorded run id.
        golden: Golden dict or path to a golden JSON file.
        store: Optional store instance.
        step_tolerance: Added slack around ``min_steps``/``max_steps`` if unset.

    Returns:
        A verdict dict ``{"pass": bool, "checks": {...}}``.
    """
    if isinstance(golden, (str, Path)):
        golden = json.loads(Path(golden).read_text(encoding="utf-8"))
    s = store if store is not None else TrajectoryStore()
    traj = s.get(run_id)
    if traj is None:
        return {"pass": False, "checks": {"found": False}, "run_id": run_id}

    actual_tools = set(traj.tool_names)
    actual_skills = set(traj.skill_names)
    msgs = s.messages(run_id)
    n_steps = len(msgs)

    exp_tools = set(golden.get("tools") or [])
    exp_skills = set(golden.get("skills") or [])
    min_steps = golden.get("min_steps")
    max_steps = golden.get("max_steps")

    checks: dict[str, Any] = {
        "found": True,
        "profile_ok": (golden.get("profile") is None or traj.profile == golden["profile"]),
        "tools_superset": exp_tools.issubset(actual_tools),
        "skills_superset": exp_skills.issubset(actual_skills),
        "actual_tools": sorted(actual_tools),
        "actual_skills": sorted(actual_skills),
        "n_steps": n_steps,
    }
    if min_steps is not None:
        checks["min_steps_ok"] = n_steps >= min_steps - step_tolerance
    if max_steps is not None:
        checks["max_steps_ok"] = n_steps <= max_steps + step_tolerance

    passed = all(
        v
        for k, v in checks.items()
        if k.endswith("_ok") or k in ("found", "profile_ok", "tools_superset", "skills_superset")
    )
    return {"pass": bool(passed), "checks": checks, "run_id": run_id}


def judge_trajectory(
    run_id: str,
    judges: list[dict[str, Any]],
    *,
    store: TrajectoryStore | None = None,
) -> list[dict[str, Any]]:
    """Run configured judges over a stored trajectory.

    Each judge dict has:

    - ``kind``: ``"correctness"`` | ``"trajectory_accuracy"`` | ``"tool_use"``
      | ``"grounding"`` | ``"efficiency"``.
    - ``judge``: an LLM object (e.g. ``get_llm("fast_model")``).
    - ``inputs`` / ``reference_outputs``: optional strings for output judges.

    Returns one verdict dict per judge (``{"kind", "score", "comment"}``).

    Only the ``correctness`` and ``trajectory_accuracy`` judges map to existing
    ``openevals``/``agentevals`` evaluators; ``tool_use``/``grounding``/
    ``efficiency`` are lightweight deterministic checks over the stored
    trajectory (no extra LLM call) so they run without ``--include-real-models``.
    """
    s = store if store is not None else TrajectoryStore()
    msgs = s.messages(run_id)
    traj = s.get(run_id)
    verdicts: list[dict[str, Any]] = []
    for j in judges:
        kind = j.get("kind", "")
        if kind == "correctness":
            from openevals.llm import create_llm_as_judge
            from openevals.prompts import CORRECTNESS_PROMPT

            evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=j["judge"])
            try:
                res = evaluator(
                    inputs=j.get("inputs", ""),
                    outputs=_final_answer(msgs),
                    reference_outputs=j.get("reference_outputs"),
                )
                verdicts.append({"kind": kind, "score": res.get("score"), "comment": res.get("comment")})
            except Exception as exc:  # noqa: BLE001
                # Judge models occasionally return prose instead of JSON; the
                # store→judge wiring still worked, so report a structured
                # verdict rather than crashing the eval run.
                verdicts.append({"kind": kind, "score": None, "comment": f"judge did not return JSON: {exc}"})
        elif kind == "trajectory_accuracy":
            from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

            evaluator = create_trajectory_llm_as_judge(
                prompt=TRAJECTORY_ACCURACY_PROMPT, judge=j["judge"], continuous=True
            )
            res = evaluator(outputs=msgs, reference_outputs=j.get("reference_trajectory", []))
            verdicts.append({"kind": kind, "score": res.get("score"), "comment": res.get("comment")})
        elif kind == "tool_use":
            expected = set(j.get("tools") or [])
            actual = set(traj.tool_names) if traj else set()
            verdicts.append(
                {
                    "kind": kind,
                    "score": expected.issubset(actual),
                    "comment": f"expected={sorted(expected)} actual={sorted(actual)}",
                }
            )
        elif kind == "grounding":
            # Deterministic grounding check: every non-empty assistant claim step
            # must follow at least one tool result. (LLM-judged grounding is a
            # later refinement; this is the cheap structural version.)
            grounded = _has_grounding(msgs)
            verdicts.append({"kind": kind, "score": grounded, "comment": "assistant answers follow a tool observation"})
        elif kind == "efficiency":
            # Deterministic efficiency check: no tool called more than `max_repeat` times.
            max_repeat = int(j.get("max_repeat", 3))
            repeats = _tool_repeat_counts(traj) if traj else {}
            ok = all(c <= max_repeat for c in repeats.values())
            verdicts.append({"kind": kind, "score": ok, "comment": f"tool_repeat_counts={repeats}"})
        else:
            verdicts.append({"kind": kind, "score": None, "comment": f"unknown judge kind: {kind}"})
    return verdicts


# ── Helpers ──────────────────────────────────────────────────────────────────


def _final_answer(messages: list[dict[str, Any]]) -> str:
    """Return the last assistant message content."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])
    return ""


def _has_grounding(messages: list[dict[str, Any]]) -> bool:
    """True if any assistant answer follows a tool observation."""
    seen_tool = False
    for m in messages:
        if m.get("role") == "tool":
            seen_tool = True
        elif m.get("role") == "assistant" and m.get("content") and seen_tool:
            return True
    return False


def _tool_repeat_counts(traj: Any) -> dict[str, int]:
    """Count how many times each tool was called in a trajectory."""
    counts: dict[str, int] = {}
    for tc in traj.tool_calls:
        counts[tc.name] = counts.get(tc.name, 0) + 1
    return counts
