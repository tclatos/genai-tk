---
name: genai-tk-core-models
description: Work on core LLM, embeddings, vector store, provider, cache, prompt, and retriever factories in genai-tk. Use when editing genai_tk/core or provider configuration.
---

# GenAI Toolkit Core Models

## Read First

- `docs/core.md`
- `docs/llm-selection.md`
- `genai_tk/core/factories/llm_factory.py`
- `genai_tk/core/factories/embeddings_factory.py`
- `genai_tk/core/factories/retriever_factory.py`
- `config/providers/llm.yaml`
- `config/providers/embeddings.yaml`

## Main Components

| Component | Code | Tests |
|---|---|---|
| LLM creation | `genai_tk/core/factories/llm_factory.py` | `tests/unit_tests/core/test_llm_factory.py` |
| Embeddings | `genai_tk/core/factories/embeddings_factory.py` | `tests/unit_tests/core/test_embeddings_factory.py` |
| Vector stores | `genai_tk/core/embeddings_store.py`, `genai_tk/core/vector_backends/` | `tests/unit_tests/core/test_embeddings_store.py` |
| Retrievers | `genai_tk/core/factories/retriever_factory.py`, `genai_tk/core/retrievers/` | `tests/unit_tests/core/test_retriever_factory.py` |
| Cache | `genai_tk/core/cache.py` | `tests/unit_tests/core/test_cache.py` |
| Prompts | `genai_tk/core/prompts.py` | `tests/unit_tests/core/test_prompts.py` |

## Patterns

- Model IDs use `model@provider` or config aliases such as `default` and `fast_model`.
- Keep provider-specific branching inside factories or provider adapters.
- Prefer YAML aliases in `config/providers/*.yaml` for new models.
- Use fake providers or the `pytest` profile for unit tests.
- Structured output should use Pydantic models and `with_structured_output` when available.

## Verification

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/core -q
uv run cli core llm "hello" --llm fast_model
```

Use real provider tests only when explicitly validating live integration behavior.
