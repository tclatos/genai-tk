---
name: core-models
description: Work on core LLM, embeddings, vector store, provider, cache, prompt, and retriever factories in genai-tk. Use when editing genai_tk/core or provider configuration.
---

# GenAI Toolkit Core Models

## Read First

- `docs/core.md`
- `docs/llm-selection.md`
- `genai_tk/core/factories/llm_factory.py`
- `genai_tk/core/factories/embeddings_factory.py`
- `genai_tk/core/factories/decision_model_factory.py`
- `genai_tk/core/factories/retriever_factory.py`
- `config/providers/llm.yaml`
- `config/providers/embeddings.yaml`
- `config/providers/decision_models.yaml`

## Main Components

| Component | Code | Tests |
|---|---|---|
| LLM creation | `genai_tk/core/factories/llm_factory.py` | `tests/unit_tests/core/test_llm_factory.py` |
| Decision Models | `genai_tk/core/factories/decision_model_factory.py`, `genai_tk/core/decision_models/` | `tests/unit_tests/core/test_decision_model.py` |
| Embeddings | `genai_tk/core/factories/embeddings_factory.py` | `tests/unit_tests/core/test_embeddings_factory.py` |
| Vector stores | `genai_tk/core/embeddings_store.py`, `genai_tk/core/vector_backends/` | `tests/unit_tests/core/test_embeddings_store.py` |
| Retrievers | `genai_tk/core/factories/retriever_factory.py`, `genai_tk/core/retrievers/` | `tests/unit_tests/core/test_retriever_factory.py` |
| Cache | `genai_tk/core/cache.py` | `tests/unit_tests/core/test_cache.py` |
| Prompts | `genai_tk/core/prompts.py` | `tests/unit_tests/core/test_prompts.py` |

## Patterns

- Model IDs use `model@provider` or config aliases such as `default` and `fast_model`.
- **Inversion of Control (IoC) Providers**:
  - Providers in `config/providers/providers.yaml` specify role-targeted classes: `chat_class`, `embeddings_class`, and `decision_class` (avoiding duplicate legacy `use:` entries).
  - Providers without chat capability (e.g. `laya` which is decision-only, or `local` which is embedding-only) omit `chat_class`.
  - Factories (`LlmFactory`, `EmbeddingsFactory`, `DecisionModelFactory`) dynamically resolve and instantiate classes via `ImportResolver`.
  - Subclassed LangChain models live in `genai_tk/core/providers/chat.py` with modular mixins in `genai_tk/core/providers/mixins.py` (e.g. `SeedHandlingMixin`, `OpenRouterRoutingMixin`, `EdenAIRoutingMixin`, `ReasoningMixin`, `EdenAIReasoningMixin`).
- Downstream applications can dynamically add custom providers or local servers (such as on-prem vLLM or Ollama) by defining top-level `providers:` in their YAML configuration.
- Prefer YAML aliases in `config/providers/*.yaml` for new models.
- Use fake providers or the `pytest` profile for unit tests.
- Persistent key-value and embedding caches use the JSON-backed `FileTreeStore` through `extra/kv_store_factory.py`; tests use `MemoryStore`. Keep the unused `py-key-value-aio[disk]` extra and DiskCache out of the dependency graph. Preserve binary values through `AsyncKeyValueByteStoreAdapter`, namespace collections, and verify persistence/deletion with [the filesystem cache regressions](../../../tests/integration_tests/security/test_filetree_cache.py). Rebuild old DiskCache caches instead of deserializing them during migration; FileTreeStore requires single-writer coordination for concurrent processes.
- Structured output should use Pydantic models and `with_structured_output` when available.

## Verification

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/core -q
uv run cli core llm "hello" --llm fast_model
```

Use real provider tests only when explicitly validating live integration behavior.
