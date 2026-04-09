# Evaluation Testing Guide

This document explains the evaluation test framework for GenAI Toolkit, built on **openevals** and **agentevals** libraries. It covers how to run, write, and monitor agent evaluations.

## Overview

The evaluation framework provides three categories of automated tests:

1. **Deterministic Trajectory Tests** — Match agent output against expected tool call sequences (no LLM needed, fast)
2. **LLM-as-Judge Tests** — Score agent outputs using a language model for semantic correctness (real API calls, slower)
3. **Multi-Turn Simulation** — Test multi-turn conversations and gather evaluator scores across turns

All tests live in `tests/eval_tests/` and are marked with the `@pytest.mark.evals` decorator.

---

## Running Evaluations

### Quick Check: Deterministic Tests Only (No API Calls)

```bash
# Run the fast trajectory-matching evaluations
make test-evals
# or equivalently:
uv run pytest tests/eval_tests/ -m "evals and not real_models" -v
```

**Result:** ~20 tests in ~0.5 seconds, no API calls needed.

### Full Suite: All Tests Including Real Models

```bash
# Run all tests, including LLM-judged evaluations (requires OpenRouter API key)
make test-evals-full
# or equivalently:
uv run pytest tests/eval_tests/ -m "evals" --include-real-models -v --timeout=120
```

**Result:** ~29 tests in ~40 seconds, uses `fast_model` (claude-haiku@openrouter) as the judge.

### Diagnostic Probe Script

For troubleshooting or exploring evaluator behavior:

```bash
# Test raw LLM latency and structured output support
uv run python scripts/evals_probe.py

# Or run specific suites
PROBE_SUITES=raw_llm,correctness uv run python scripts/evals_probe.py

# Available suites: raw_llm, correctness, conciseness, trajectory_match, trajectory_llm_judge, multiturn
PROBE_MODEL=gpt-4@openai PROBE_SUITES=correctness uv run python scripts/evals_probe.py
```

---

## Architecture

### Test Organization

```
tests/eval_tests/
├── conftest.py              # Shared fixtures (eval_agent, judge_llm, tools)
├── test_llm_judged.py       # openevals LLM-as-judge tests (7 real-model tests)
├── test_trajectory_match.py # agentevals trajectory matching (15 deterministic tests)
└── test_multiturn.py        # Multi-turn simulation and LangchainAgent integration
```

### Key Fixtures

**`eval_agent`** — A `LangchainAgent` with:
- LLM: `parrot_local@fake` (zero cost, instant)
- Tools: `calculator`, `echo` (simple, deterministic)
- Type: `react` (reasoning + action loop)

**`judge_llm`** — The configured `fast_model` (claude-haiku@openrouter by default):
- Skipped automatically if `--include-real-models` is not set
- Used by all LLM-as-judge evaluators

**`calculator`, `echo`** — Minimal tools for deterministic testing:
- `calculator(expression: str)` → evaluates arithmetic (e.g., "2+3" → "5.0")
- `echo(message: str)` → returns the message unchanged

---

## Writing Tests

### 1. Deterministic Trajectory Tests (No LLM)

Use these to verify an agent calls the right tools with the right arguments.

**Example: Verify agent calls calculator for arithmetic**

```python
from agentevals.trajectory.match import create_trajectory_match_evaluator

def test_agent_uses_calculator_for_math():
    """Verify agent calls the calculator tool for arithmetic."""
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",  # Allow extra turn steps
        tool_args_match_mode="ignore",     # Don't check order of args
    )
    
    # Expected: agent should call calculator
    expected_trajectory = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": "{}"}}]
        },
    ]
    
    # Actual agent trajectory
    actual_trajectory = [
        {"role": "user", "content": "What is 5 * 3?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "calculator", "arguments": '{"expression": "5 * 3"}'}}]},
        {"role": "tool", "content": "15.0"},
        {"role": "assistant", "content": "5 * 3 equals 15."},
    ]
    
    result = evaluator(outputs=actual_trajectory, reference_outputs=expected_trajectory)
    assert result["score"], f"Trajectory evaluator failed: {result['feedback']}"
```

**Available trajectory modes:**
- `"superset"` — Actual must contain at least the expected steps (allows extra steps)
- `"subset"` — Expected must contain at least the actual steps (agent can call fewer)
- `"strict"` — Exact match only

**Available args modes:**
- `"exact"` — Arguments must match exactly
- `"ignore"` — Accept any arguments (only check tool name)

### 2. LLM-as-Judge Tests (Requires API Key)

Use these to score agent outputs for semantic quality using a language model.

**Example: Score answer correctness**

```python
import pytest
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(120)
def test_correct_answer_is_scored_high(judge_llm) -> None:
    """A correct answer should receive a high score from CORRECTNESS_PROMPT."""
    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge_llm)
    
    result = evaluator(
        inputs="What is 17 * 3?",
        outputs="17 multiplied by 3 equals 51.",
        reference_outputs="51",
    )
    
    score = result.get("score")
    assert score, f"Expected PASS but got score={score!r}. Reasoning: {result.get('comment')}"
```

**Available openevals prompts:**
- `CORRECTNESS_PROMPT` — Did the answer match the reference output?
- `CONCISENESS_PROMPT` — Is the answer brief and to-the-point?
- `RELEVANCE_PROMPT` — Is the answer relevant to the input?
- `FAITHFULNESS_PROMPT` — Is the answer grounded in the provided context?

**Available agentevals prompts:**
- `TRAJECTORY_ACCURACY_PROMPT` — Was the trajectory of tool calls correct and complete?

### 3. Multi-Turn Simulation Tests

Use these to test end-to-end agent behavior across multiple conversation turns.

**Example: Test multi-turn agent conversation**

```python
import asyncio
import concurrent.futures
from openevals.simulators import run_multiturn_simulation
from genai_tk.agents.langchain.langchain_agent import LangchainAgent

@pytest.mark.evals
@pytest.mark.real_models
def test_agent_handles_multiturn_conversation(judge_llm):
    """Agent maintains context and produces correct answers across turns."""
    agent = LangchainAgent(llm="fast_model", agent_type="react", tools=[calculator, echo])
    
    def agent_app(message: dict, **kwargs) -> dict:
        """Wrapper to call async agent from sync simulator."""
        content = message.get("content", "")
        
        def _run():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(agent.arun(content))
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(_run).result(timeout=60)
        
        return {"role": "assistant", "content": response}
    
    def check_answers(*, outputs, reference_outputs=None, **kwargs):
        """Custom evaluator: check for expected numeric answers in trajectory."""
        trajectory = outputs if isinstance(outputs, list) else outputs.get("messages", [])
        text = " ".join(
            str(m.get("content", ""))
            for m in trajectory
            if m.get("role") == "assistant"
        )
        has_96 = "96" in text
        has_192 = "192" in text
        return {
            "key": "correct_answers",
            "score": has_96 and has_192,
            "comment": f"96={has_96}, 192={has_192}",
        }
    
    result = run_multiturn_simulation(
        app=agent_app,
        user=["What is 12 * 8?", "What is 96 * 2?"],
        max_turns=2,
        trajectory_evaluators=[check_answers],
    )
    
    eval_results = result.get("evaluator_results", [])
    for er in eval_results:
        assert er.get("score"), f"Evaluator '{er.get('key')}' failed: {er.get('comment')}"
```

### Assertion Helpers

Use these to provide informative failure messages:

```python
def _assert_pass(result: dict, label: str) -> None:
    """Assert evaluator result is a pass (truthy score)."""
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning)"
    assert score, f"[{label}] Expected PASS but got {score!r}. Reasoning: {reasoning}"

def _assert_fail(result: dict, label: str) -> None:
    """Assert evaluator result is a fail (falsy score)."""
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning)"
    assert not score, f"[{label}] Expected FAIL but got {score!r}. Reasoning: {reasoning}"
```

---

## Configuration

### Environment Variables

```bash
# Set the model to use for LLM-judged evaluations (default: fast_model)
JUDGE_MODEL=gpt-4@openai uv run pytest tests/eval_tests/ --include-real-models

# Set timeout per test (default: 120 seconds)
uv run pytest tests/eval_tests/ --timeout=60
```

### Config Files

The `judge_llm` fixture uses the project's configured `fast_model`:

```yaml
# config/baseline.yaml
llm:
  models:
    fast_model: claude-haiku@openrouter
    cheap_model: claude-haiku@openrouter
```

To use a different judge for evaluations, edit this config or override at runtime:

```bash
# Override via CLI (requires environment variable export)
OPENROUTER_API_KEY=... JUDGE_MODEL=gpt-4-mini@openai make test-evals-full
```

---

## Design Notes

### Why Deterministic Tests First?

- **No API calls** — Safe to run in CI/CD without credentials
- **Fast feedback** — Verify tool-calling behavior in milliseconds
- **Regression detection** — Catch broken agent scaffolding early
- **Complement LLM tests** — LLM tests are slow and potentially flaky; deterministic tests are reliable

### Why claude-haiku@openrouter for judge_llm?

- **Mature model** — More stable and predictable than experimental models
- **Fast inference** — ~1-2 seconds per evaluation
- **Cost-effective** — Cheap per-token pricing via OpenRouter
- **Structured output support** — Reliably formats JSON responses

### Timeout Strategy

- **Per-test timeout: 120 seconds** — Prevents hung API calls from blocking the test suite
- **Deterministic tests: instant** — No timeout needed, but marked for consistency
- **LLM-judged tests: 10-30 seconds** — Typical API latency to claude-haiku
- **Multi-turn tests: 30-60 seconds** — Agent reasoning + multiple API calls

### Handling Flakiness

LLM-as-judge tests can be flaky (different models, prompt variations, temperature). To mitigate:

1. **Isolation** — Each test is independent; failure of one doesn't affect others
2. **Clear assertions** — Always print judge reasoning on failure for debugging
3. **Semantic thresholds** — Don't assert on exact scores; use `bool(score)` for binary results
4. **Opt-in** — Real-model tests are skipped by default; only run with `--include-real-models`

---

## Common Patterns

### Create a Custom Evaluator

```python
def my_custom_evaluator(*, outputs, reference_outputs=None, **kwargs):
    """Custom evaluator: check for specific text in output."""
    text = outputs if isinstance(outputs, str) else outputs.get("content", "")
    found = "success" in text.lower()
    return {
        "key": "success_indicator",
        "score": found,
        "comment": "Found 'success' in output" if found else "No 'success' found",
        "metadata": None,
    }

# Use in a test
evaluator_results = run_multiturn_simulation(
    app=agent_app,
    user=["Do something successful"],
    max_turns=1,
    trajectory_evaluators=[my_custom_evaluator],
)
```

### Test With Different LLMs

```python
@pytest.mark.parametrize("model", ["fast_model", "gpt-4@openai"])
def test_with_different_models(model):
    agent = LangchainAgent(llm=model, agent_type="react", tools=[calculator])
    # ... rest of test
```

### Debug Trajectories

```python
result = run_multiturn_simulation(...)
trajectory = result.get("trajectory", [])

# Print the full trajectory for debugging
for i, msg in enumerate(trajectory):
    role = msg.get("role", "?")
    content = str(msg.get("content", ""))[:100]
    print(f"{i}: [{role}] {content}")
```

---

## Suggested Improvements

### Short-Term (Low Effort)

1. **Fixture for custom evaluators** — Create a parameterizable evaluator factory fixture
   ```python
   @pytest.fixture
   def custom_evaluator(text_to_find: str):
       def evaluator(*, outputs, **kwargs):
           found = text_to_find in str(outputs)
           return {"key": "custom", "score": found}
       return evaluator
   ```

2. **Metrics tracking** — Log eval metrics (latency, cost) to a results file
   ```python
   # After test runs, write CSV with: test_name, duration, model, score
   ```

3. **Pytest hooks for snapshot testing** — Save expected trajectories, auto-compare
   ```python
   # If trajectory differs from snapshot, show diff clearly
   ```

### Medium-Term (Moderate Effort)

4. **Parallel execution** — Run independent evaluations in parallel (currently sequential)
   ```bash
   pytest tests/eval_tests/ -n auto  # Requires pytest-xdist
   ```

5. **Flakiness retries** — Auto-retry flaky LLM-judged tests with backoff
   ```python
   @pytest.mark.flaky(reruns=2)
   def test_llm_judged(...):
       ...
   ```

6. **Cost tracking** — Aggregate API spend per test run
   ```bash
   # Display: "Ran 29 evals for $0.14 (claude-haiku @ 0.005/M tokens)"
   ```

7. **Integration with LangSmith** — Auto-log all eval traces to LangSmith for analysis
   ```python
   # Already has langsmith in dependencies; just needs config
   ```

### Long-Term (Major Effort)

8. **Custom evaluators library** — Build reusable evaluator components
   ```python
   from genai_tk.evals import evaluators
   
   # Pre-built: exact_match, contains_text, code_valid, json_valid, etc.
   code_quality = evaluators.code_quality_check(judge_llm)
   ```

9. **Eval dashboard** — Web UI to view eval results, trends, failure cases
   - Historical trend charts (accuracy over time)
   - Failure case categorization (false negatives, timeouts, etc.)
   - A/B comparison (model A vs model B on same evals)

10. **Auto-generation** — Generate test cases from agent logs
    - Train on successful interactions, generate negative test cases
    - Fuzz agent with edge cases derived from test distribution

11. **Continuous evals** — Background job to run evals on every commit
    - Warn if accuracy drops below threshold
    - Alert on new flaky tests

---

## Troubleshooting

### Tests are stuck/timing out

**Symptom:** Pytest waits forever on a test.

**Cause:** LLM-as-judge test waiting for API response that never completes.

**Solution:**
```bash
# Set explicit timeout
pytest tests/eval_tests/ --timeout=30

# Or run deterministic tests only
make test-evals  # Skips real_models
```

### Evaluator returns wrong schema

**Symptom:** `KeyError: 'score'` or similar.

**Cause:** Evaluator returned dict without expected keys.

**Solution:**
```python
# Always return a dict with: key, score, comment, metadata
def evaluator(*, outputs, **kwargs):
    return {
        "key": "my_eval",
        "score": True,  # Must be bool or int (truthy/falsy)
        "comment": "Why this score",
        "metadata": None,  # Optional, for custom data
    }
```

### "OPENROUTER_API_KEY not found"

**Symptom:** Tests fail with API key error.

**Cause:** Missing environment variable or config.

**Solution:**
```bash
# Set API key in environment
export OPENROUTER_API_KEY=sk_...
make test-evals-full

# Or set in .env file
echo "OPENROUTER_API_KEY=sk_..." >> .env
```

### Judge is scoring everything wrong

**Symptom:** Correct answers are failing; incorrect answers are passing.

**Cause:** Judge prompt mismatch or model misunderstanding.

**Solution:**
```bash
# Use the diagnostic probe to test judge directly
uv run python scripts/evals_probe.py

# Check judge's reasoning
result = evaluator(inputs="...", outputs="...", reference_outputs="...")
print(result.get("comment"))  # Read the reasoning
```

---

## References

- **openevals** — https://github.com/openevals/evals
- **agentevals** — https://github.com/langchain-ai/agentevals
- **LangGraph Trajectory Docs** — https://langgraph.dev/concepts/trajectories
- **OpenRouter Models** — https://openrouter.ai/models

---

## Contributing New Tests

### Checklist

- [ ] Test is marked with `@pytest.mark.evals`
- [ ] Real-model tests marked with `@pytest.mark.real_models`
- [ ] Real-model tests include `@pytest.mark.timeout(120)`
- [ ] Docstring explains what is being tested
- [ ] Assertions have clear failure messages
- [ ] Tools used are deterministic (`calculator`, `echo`, or custom mocks)
- [ ] Test passes locally before pushing: `make test-evals` or `make test-evals-full`

### Example PR

When adding a new eval test, include:

1. **Test file** in `tests/eval_tests/`
2. **Brief docstring** explaining the eval
3. **Clear assertions** with debug info
4. **Comments** on non-obvious evaluator choices

```python
@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(120)
def test_agent_handles_ambiguous_input(judge_llm) -> None:
    """Agent should ask for clarification when input is ambiguous.
    
    Verifies that the agent uses clarification language when the user's
    intent is unclear, rather than guessing.
    """
    evaluator = create_llm_as_judge(
        prompt=RELEVANCE_PROMPT,  # Is the response relevant?
        judge=judge_llm,
    )
    
    # ... rest of test
```

---

## Summary

The evaluation framework provides:

✅ **Fast deterministic tests** for tool-calling correctness  
✅ **LLM-judged evals** for quality assessment  
✅ **Multi-turn simulation** for conversation testing  
✅ **Opt-in real models** to avoid accidental API calls  
✅ **Timeout protection** to prevent hangs  
✅ **Clear failure diagnostics** with judge reasoning  

Use this as a foundation for building production-grade agent evaluations.
