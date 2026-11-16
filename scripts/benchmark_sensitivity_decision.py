"""Benchmark sensitivity classification latency as a function of message length.

Compares two scorers usable by :class:`~genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterMiddleware`:

1. ``DecisionModelSensitivityScorer`` backed by Laya (local, in-process router, CPU)
2. ``DefaultSensitivityScorer`` (regex + keywords + Presidio baseline)

Usage:

    ```bash
    uv run scripts/benchmark_sensitivity_decision.py
    uv run scripts/benchmark_sensitivity_decision.py --repeats 50 --json /tmp/results.json
    ```
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

FILLER = (
    "The team reviewed the quarterly roadmap and agreed to postpone the migration to next quarter. "
    "Please prepare a short summary of the open action items before the next status meeting. "
    "Infrastructure costs stayed flat while traffic grew moderately across all regions. "
)

PII_SNIPPET = "My login is john.doe@corp.com and my password is hunter2; my IBAN is FR7630006000011234567890189."


def make_message(length: int) -> str:
    """Build a realistic message of *length* characters with an embedded PII snippet."""
    if length <= len(PII_SNIPPET):
        return PII_SNIPPET[:length]
    half = (length - len(PII_SNIPPET)) // 2
    filler = FILLER * (half // len(FILLER) + 1)
    message = filler[:half] + " " + PII_SNIPPET + " " + filler[half:]
    return message[:length]


def _percentile(sorted_values: list[float], pct: float) -> float:
    index = min(int(pct / 100 * (len(sorted_values) - 1)), len(sorted_values) - 1)
    return sorted_values[index]


def bench_scorer(scorer: Callable[[str], Any], text: str, repeats: int, warmup: int) -> dict[str, Any]:
    """Time *repeats* classifications of *text* after *warmup* calls; return latency stats in ms."""
    sensitive_count = 0
    for _ in range(warmup):
        result = scorer(text)
        sensitive_count += int(bool(getattr(result, "is_sensitive", False)))

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = scorer(text)
        timings.append((time.perf_counter() - start) * 1000.0)
        sensitive_count += int(bool(getattr(result, "is_sensitive", False)))

    ordered = sorted(timings)
    return {
        "mean_ms": round(statistics.fmean(ordered), 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(_percentile(ordered, 95), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "sensitive_ratio": round(sensitive_count / (repeats + warmup), 2),
    }


def _cpu_model() -> str:
    if platform.system() == "Linux":
        for line in _read_cpuinfo_lines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _read_cpuinfo_lines() -> list[str]:
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        return handle.readlines()


def main() -> None:
    """Run the latency benchmark and print a markdown table of results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30, help="Timed classifications per length bucket")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warmup classifications per bucket")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[25, 100, 250, 500, 1000, 2000, 4000],
        help="Message lengths in characters",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional path to dump raw results as JSON")
    args = parser.parse_args()

    from genai_tk.agents.langchain.middleware.decision_model_scorer import DecisionModelSensitivityScorer
    from genai_tk.extra.nlp.classifiers import DefaultScorerConfig, DefaultSensitivityScorer

    print(f"CPU: {_cpu_model()}")
    print(f"Repeats per bucket: {args.repeats} (warmup {args.warmup})\n")

    load_start = time.perf_counter()
    laya = DecisionModelSensitivityScorer(model_id="laya@laya", model_kwargs={"device": "cpu", "preload": True})
    laya_load_s = time.perf_counter() - load_start

    try:
        baseline = DefaultSensitivityScorer(DefaultScorerConfig())
        baseline_note = "default config"
    except Exception as exc:  # pragma: no cover - environment dependent
        from genai_tk.extra.nlp import PresidioDetectorConfig

        baseline = DefaultSensitivityScorer(DefaultScorerConfig(detector=PresidioDetectorConfig(enable_spacy=False)))
        baseline_note = f"spacy disabled ({exc})"

    print(f"Laya model load (in-process router, first init): {laya_load_s:.1f}s")
    print(f"Baseline scorer: DefaultSensitivityScorer ({baseline_note})\n")

    results: dict[str, Any] = {
        "cpu": _cpu_model(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "laya_load_s": round(laya_load_s, 2),
        "buckets": {},
    }

    header = (
        "| chars | Laya median (p95) ms | baseline median (p95) ms | Laya / baseline |"
        " Laya sensitive | baseline sensitive |"
    )
    print(header)
    print("|---|---|---|---|---|---|")

    for length in args.lengths:
        text = make_message(length)
        row: dict[str, Any] = {}
        try:
            laya_stats = bench_scorer(laya.assess, text, args.repeats, args.warmup)
        except Exception as exc:
            laya_stats = {"error": str(exc)}
        try:
            base_stats = bench_scorer(baseline.assess, text, args.repeats, args.warmup)
        except Exception as exc:
            base_stats = {"error": str(exc)}

        row["laya"] = laya_stats
        row["baseline"] = base_stats
        results["buckets"][str(length)] = row

        if "error" in laya_stats or "error" in base_stats:
            print(f"| {length} | {laya_stats.get('error', '')} | {base_stats.get('error', '')} | | | |")
            continue

        laya_median = float(laya_stats["median_ms"])
        base_median = float(base_stats["median_ms"])
        ratio = laya_median / base_median if base_median > 0 else float("inf")
        print(
            f"| {length} "
            f"| {laya_stats['median_ms']:.1f} ({laya_stats['p95_ms']:.1f}) "
            f"| {base_stats['median_ms']:.1f} ({base_stats['p95_ms']:.1f}) "
            f"| {ratio:.1f}x "
            f"| {laya_stats['sensitive_ratio']:.0%} "
            f"| {base_stats['sensitive_ratio']:.0%} |"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nRaw results written to {args.json}")


if __name__ == "__main__":
    main()
