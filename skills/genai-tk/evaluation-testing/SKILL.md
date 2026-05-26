---
name: genai-tk-evaluation-testing
description: Add or debug pytest unit, integration, and LLM evaluation tests for genai-tk, including fake models, real-model gates, and trajectory tests.
---

# GenAI Toolkit Evaluation And Testing

## Read First

- `docs/TESTING_GUIDE.md`
- `docs/evaluation_testing.md`
- `tests/conftest.py`
- `tests/utils/`
- `pyproject.toml` dependency groups `dev` and `evals`

## Test Layout

| Tier | Path | Use for |
|---|---|---|
| Unit | `tests/unit_tests/` | Fast deterministic behavior and config parsing |
| Integration | `tests/integration_tests/` | CLI wiring, real services, browser, end-to-end flows |
| Eval | `tests/eval_tests/` | LLM quality, trajectory, multiturn behavior |
| Utilities | `tests/utils/` | Shared fake data, factories, constants |

## Patterns

- Use `GENAITK_PROFILE=pytest` for deterministic tests.
- Use fake LLM/embedding config for unit tests.
- Mark real model tests explicitly and require opt-in flags.
- Use `pytest.importorskip()` for optional packages.
- Prefer narrow test files next to the domain being changed.

## Commands

```bash
uv run pytest tests/unit_tests -q
uv run pytest tests/integration_tests -q
uv run pytest tests/eval_tests -q
make test-unit
make test-integration
```

## Avoid

- Do not call live LLM providers from unit tests.
- Do not require Docker for unit tests.
- Do not remove flaky real-service tests without replacing their coverage or marking them with a reason.
