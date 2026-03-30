"""Diagnostic script for openevals / agentevals integration.

Runs each evaluator call in isolation, prints timing and full results so that
failures or hangs can be diagnosed without pytest overhead.

Usage:
    uv run python scripts/evals_probe.py

Optional env vars:
    PROBE_MODEL  — model ID to use as judge   (default: fast_model)
    PROBE_SUITES — comma-separated list of suites to run, or "all"
                   choices: raw_llm, correctness, conciseness,
                            trajectory_match, trajectory_llm_judge,
                            multiturn
                   (default: all)

Examples:
    # Run everything with fast_model
    uv run python scripts/evals_probe.py

    # Run only the synchronous (no-LLM) suites to confirm the library works
    PROBE_SUITES=trajectory_match,multiturn uv run python scripts/evals_probe.py

    # Run LLM-judge suites with a specific model
    PROBE_MODEL=fast_model PROBE_SUITES=correctness,conciseness uv run python scripts/evals_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# ── config ──────────────────────────────────────────────────────────────────
JUDGE_MODEL = os.environ.get("PROBE_MODEL", "fast_model")
_suites_env = os.environ.get("PROBE_SUITES", "all")
RUN_ALL = _suites_env == "all"
SUITES = set(_suites_env.split(",")) if not RUN_ALL else set()

# ── helpers ──────────────────────────────────────────────────────────────────

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

_results: list[tuple[str, bool, float, str]] = []  # (suite, passed, elapsed, detail)


def _enabled(suite: str) -> bool:
    return RUN_ALL or suite in SUITES


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def run(label: str, fn, *args, **kwargs):
    """Call ``fn(*args, **kwargs)``, print result + timing, record pass/fail."""
    print(f"\n  ▶ {label} … ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        # Pretty-print dict results; truncate long strings
        if isinstance(result, dict):
            pretty = json.dumps(
                {k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v) for k, v in result.items()},
                indent=2,
            )
        else:
            pretty = str(result)
        print(f"{GREEN}OK{RESET} ({elapsed:.2f}s)")
        print(f"     {pretty}")
        _results.append((label, True, elapsed, ""))
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        import traceback

        detail = traceback.format_exc()
        print(f"{RED}FAIL{RESET} ({elapsed:.2f}s): {type(exc).__name__}: {exc}")
        print(detail)
        _results.append((label, False, elapsed, str(exc)))
        return None


# ── bootstrap config ─────────────────────────────────────────────────────────
from genai_tk.utils.config_mngr import global_config  # noqa: E402

global_config().select_config("pytest")

# ── 1. raw LLM latency ───────────────────────────────────────────────────────
if _enabled("raw_llm"):
    section("1. Raw LLM latency")

    from langchain_core.messages import HumanMessage

    from genai_tk.core.llm_factory import get_llm

    def _raw_invoke():
        m = get_llm(JUDGE_MODEL)
        r = m.invoke([HumanMessage(content="Reply with exactly the word OK and nothing else.")])
        return {"content": r.content, "model": getattr(m, "model_name", None) or getattr(m, "model", None)}

    run("raw invoke (fast_model)", _raw_invoke)

    def _structured_output():
        """Check whether the model supports with_structured_output (used by openevals)."""
        m = get_llm(JUDGE_MODEL)
        schema = {
            "title": "score",
            "type": "object",
            "properties": {
                "score": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        }
        bound = m.with_structured_output(schema)
        r = bound.invoke(
            [HumanMessage(content="Is the sky blue? Answer with score=true and a one-sentence reasoning.")]
        )
        return r

    run("with_structured_output (used by openevals)", _structured_output)

# ── helper: build a json_mode-wrapped judge ──────────────────────────────────


def _make_judge():
    """Return the configured JUDGE_MODEL for use as an openevals judge."""
    from genai_tk.core.llm_factory import get_llm

    return get_llm(JUDGE_MODEL)


# ── 2. openevals correctness ─────────────────────────────────────────────────
if _enabled("correctness"):
    section("2. openevals — correctness judge")

    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    judge = _make_judge()
    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge)

    run(
        "correct answer should PASS (17*3=51)",
        evaluator,
        inputs="What is 17 * 3?",
        outputs="17 multiplied by 3 equals 51.",
        reference_outputs="51",
    )

    run(
        "wrong answer should FAIL (17*3=60)",
        evaluator,
        inputs="What is 17 * 3?",
        outputs="17 multiplied by 3 equals 60.",
        reference_outputs="51",
    )

# ── 3. openevals conciseness ─────────────────────────────────────────────────
if _enabled("conciseness"):
    section("3. openevals — conciseness judge")

    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CONCISENESS_PROMPT

    judge = _make_judge()
    evaluator = create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge)

    run(
        "concise answer should PASS",
        evaluator,
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris.",
    )

    verbose = (
        "That is a wonderful question! Let me provide you with a very comprehensive "
        "and detailed answer. Paris, which is absolutely the capital of France, is a "
        "major European city and a global center for art, fashion, and gastronomy. "
        "So to directly answer your simple question: the capital of France is Paris. "
        "I sincerely hope that helps!"
    )
    run(
        "verbose answer should FAIL",
        evaluator,
        inputs="What is the capital of France?",
        outputs=verbose,
    )

# ── 4. agentevals trajectory match (no LLM) ──────────────────────────────────
if _enabled("trajectory_match"):
    section("4. agentevals — trajectory match (deterministic, no LLM)")

    from agentevals.trajectory.match import create_trajectory_match_evaluator

    superset_eval = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")

    calculator_traj = [
        {"role": "user", "content": "What is 17 * 3?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "calculator", "arguments": "{}"}}]},
        {"role": "tool", "content": "51.0"},
        {"role": "assistant", "content": "17 * 3 equals 51."},
    ]
    reference = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "calculator", "arguments": "{}"}}]},
    ]

    run(
        "superset match — calculator called (should PASS)",
        superset_eval,
        outputs=calculator_traj,
        reference_outputs=reference,
    )

    no_tool_traj = [
        {"role": "user", "content": "What is 17 * 3?"},
        {"role": "assistant", "content": "17 * 3 is 51."},
    ]
    run(
        "superset match — no tool called (should FAIL)",
        superset_eval,
        outputs=no_tool_traj,
        reference_outputs=reference,
    )

    from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

    run(
        "graph strict match — same steps (should PASS)",
        graph_trajectory_strict_match,
        outputs={"steps": ["__start__", "agent", "tools", "agent"]},
        reference_outputs={"steps": ["__start__", "agent", "tools", "agent"]},
    )

    run(
        "graph strict match — different steps (should FAIL)",
        graph_trajectory_strict_match,
        outputs={"steps": ["__start__", "agent"]},
        reference_outputs={"steps": ["__start__", "agent", "tools", "agent"]},
    )

# ── 5. agentevals trajectory LLM judge ───────────────────────────────────────
if _enabled("trajectory_llm_judge"):
    section("5. agentevals — trajectory LLM judge")

    from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

    judge = _make_judge()
    traj_judge = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        judge=judge,
        continuous=True,
    )

    good_traj = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": json.dumps({"expression": "15 * 4"})}}],
        },
        {"role": "tool", "content": "60.0"},
        {"role": "assistant", "content": "15 * 4 equals 60."},
    ]
    ref_traj = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": json.dumps({"expression": "15 * 4"})}}],
        },
        {"role": "tool", "content": "60.0"},
        {"role": "assistant", "content": "60"},
    ]
    run(
        "trajectory judge — correct trajectory (score should be >= 0.5)",
        traj_judge,
        outputs=good_traj,
        reference_outputs=ref_traj,
    )

    bad_traj = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "echo", "arguments": json.dumps({"message": "15 * 4"})}}],
        },
        {"role": "tool", "content": "15 * 4"},
        {"role": "assistant", "content": "The result is 15 * 4."},
    ]
    run(
        "trajectory judge — wrong tool (score should be < 0.8)",
        traj_judge,
        outputs=bad_traj,
        reference_outputs=ref_traj,
    )

# ── 6. multiturn simulation (no LLM) ─────────────────────────────────────────
if _enabled("multiturn"):
    section("6. openevals — multi-turn simulation (deterministic)")

    from openevals.simulators import run_multiturn_simulation

    turn_idx = [0]
    responses = ["The answer is 42.", "You're welcome!"]

    def scripted_app(msg: dict[str, Any], *, thread_id: str | None = None, **kw) -> dict[str, Any]:
        idx = min(turn_idx[0], len(responses) - 1)
        turn_idx[0] += 1
        return {"role": "assistant", "content": responses[idx]}

    def _run_sim():
        return run_multiturn_simulation(
            app=scripted_app,
            user=["What is the answer to life?", "Thanks!"],
            max_turns=2,
        )

    result = run("2-turn simulation with scripted app", _run_sim)
    if result:
        traj = result.get("trajectory", [])
        print(f"     Trajectory length: {len(traj)} messages")
        for m in traj:
            role = m.get("role", "?") if isinstance(m, dict) else getattr(m, "role", "?")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            print(f"       [{role}] {str(content)[:80]}")

# ── summary ───────────────────────────────────────────────────────────────────
section("Summary")
passed = sum(1 for _, ok, _, _ in _results if ok)
total = len(_results)
col = GREEN if passed == total else (YELLOW if passed > 0 else RED)
print(f"\n  {col}{passed}/{total} checks passed{RESET}\n")
for label, ok, elapsed, detail in _results:
    icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    timing = f"({elapsed:.2f}s)"
    print(f"  {icon} {label} {timing}")
    if not ok:
        print(f"      {RED}{detail[:120]}{RESET}")

print()
sys.exit(0 if passed == total else 1)
