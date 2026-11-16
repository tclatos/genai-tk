# Benchmark: Sensitivity Routing with a Decision Model (Laya, local CPU)

Classification latency of `SensitivityRouterMiddleware`'s content scorer as a function of message length, comparing:

1. **Decision model** — new `DecisionModelSensitivityScorer` backed by Laya (`laya@laya`), in-process router on CPU.
2. **Baseline** — `DefaultSensitivityScorer` (regex + keyword groups + banwords + Presidio), the previous default.

## Setup

- Machine: Intel Core i7-1265U (12th gen, laptop CPU, no GPU), WSL2/Ubuntu, Python 3.12, torch 2.15.0, laya 0.4.35
- Method: 5 warmup + 30 timed `assess()` calls per bucket with `time.perf_counter`; Laya route models preloaded (`preload=True`) and Hugging Face hub offline (`HF_HUB_OFFLINE=1`) so timings reflect steady-state inference only
- Messages: neutral filler text with an embedded PII snippet (email, password, IBAN), truncated/padded to the target length
- Reproduce: `uv run scripts/benchmark_sensitivity_decision.py --repeats 30 --json /tmp/sensitivity_bench.json`

## Results

Latency in milliseconds (median, with p95 in parentheses):

| chars | Laya (decision model) | DefaultSensitivityScorer | Laya / baseline |
|---|---|---|---|
| 25 | 226.3 (248.8) | 3.8 (5.2) | 59× |
| 100 | 707.6 (767.1) | 4.2 (4.8) | 168× |
| 250 | 855.4 (1062.0) | 8.0 (9.2) | 107× |
| 500 | 1020.2 (1075.7) | 8.0 (8.8) | 128× |
| 1000 | 1562.0 (1743.0) | 12.9 (15.3) | 121× |
| 2000 | 2108.6 (2625.3) | 20.5 (23.3) | 103× |
| 4000 | 3350.6 (3881.9) | 31.7 (72.4) | 106× |

Per-message cost of routing to the safe LLM when the scorer flags content = scorer latency (the safe-LLM substitution itself is a pointer swap).

## Findings

1. **Laya scales linearly with message length at ~0.79 ms/char on this CPU** — 226 ms for a one-liner, 3.4 s for a 4 000-character message. The p95 tracks the median closely once routes are preloaded, so latency is predictable. This is far from the ~100 ms CPU figure quoted for short inputs; longer messages dominate.
2. **The regex/Presidio baseline is 60–170× faster** (4–32 ms over the same range) because its cost is driven by pattern matching, not by encoding the full text through a transformer.
3. **Model load is not free**: ~12 s to instantiate the router with all three routes (english, multilingual, typed-decisions, ~843 MB of checkpoints on first download). With `preload=False`, route models load lazily *inside* the first predictions, producing multi-second spikes — always enable `preload=True` in services.
4. **Quality caveat — verdicts were non-monotonic with generic instructions.** The embedded PII snippet was flagged 100 % at 25, 100, 1000–4000 chars but **missed at 250 and 500 chars** (p(sensitive) < 0.5), while the pattern-based baseline caught it 100 % from 100 chars up. Laya also emitted a checkpoint calibration warning (confidence uncalibrated on some entries). The decision model's semantic judgment is therefore **not yet a drop-in replacement** for pattern-based PII detection without instruction tuning.
5. Laya adds value on *semantic* sensitivity (context, implied confidentiality, non-Latin languages) that regex/Presidio cannot see.

## Recommendation

Treat the two scorers as complementary layers rather than alternatives:

- Keep `DefaultSensitivityScorer` for deterministic PII/secret patterns (it pays for itself in <1 % of one Laya call).
- Enable `decision_model` for the semantic layer, or combine both (route if **either** flags) — the combined cost is still ~1–3 s per message on CPU, which is acceptable only for low-traffic, high-safety agents. For interactive agents on CPU, prefer shorter messages or run Laya on GPU.
- Tune the `Noul` instructions (via `DecisionModelSensitivityScorer(instructions=...)`) with explicit positive/negative examples and re-evaluate before production use — the generic prompt missed the PII snippet at mid-length messages.

## Related

- Implementation: `genai_tk/agents/langchain/middleware/decision_model_scorer.py`
- Middleware docs: [middleware-pii-and-routing.md](middleware-pii-and-routing.md)
- Decision models design: [design/04_decision_models_system1.md](design/04_decision_models_system1.md)
- Benchmark script: `scripts/benchmark_sensitivity_decision.py`
