---
name: linkml-structured-extraction
description: Work on LinkML structured extraction, Pydantic v2 code generation, SchemaView metadata, CLI commands (cli linkml), and Prefect workflow integration in genai-tk.
---

# GenAI Toolkit LinkML Structured Extraction

## Read First

- `docs/linkml.md`
- `genai_tk/extra/structured/linkml_contract.py`
- `genai_tk/extra/structured/linkml_processor.py`
- `genai_tk/extra/structured/commands_linkml.py`
- `genai_tk/workflow/prefect/flows/linkml_flow.py`

## Code Map

| Concern | Paths |
|---|---|
| Contract & SchemaView | `genai_tk/extra/structured/linkml_contract.py` |
| Processor & Recovery | `genai_tk/extra/structured/linkml_processor.py` |
| Configuration Model | `genai_tk/extra/structured/config.py` |
| CLI Commands (`cli linkml`) | `genai_tk/extra/structured/commands_linkml.py` |
| Workflow Flow | `genai_tk/workflow/prefect/flows/linkml_flow.py` |
| Unit Tests | `tests/unit_tests/extra/test_linkml_contract.py`, `tests/unit_tests/extra/test_pydantic_processor.py`, `tests/unit_tests/workflow/prefect/flows/test_structured_flow.py` |
| Integration Tests | `tests/integration_tests/test_commands_structured.py` |

## Key Patterns

1. **LinkML as Schema Authority**:
   LinkML YAML files define classes, slots, ranges, and enums with permissible-value descriptions.
2. **Build-Time Pydantic Generation**:
   Generate Pydantic v2 code with `cli linkml generate <schema.yaml> --out <target.py>`. Never hand-edit generated models.
3. **CI Drift Detection**:
   Verify generated code matches schema via `cli linkml validate-drift <schema.yaml> <target.py>` or `just linkml-validate-drift`.
4. **LinkML Processor**:
   - `LinkMLProcessor` invokes LangChain chat models via `llm.with_structured_output(model, include_raw=True)`.
   - Fast path returns validated parsed model.
   - Recovery path uses `json_repair` in schema mode on malformed or truncated raw JSON.
   - Strict fallback prompts with JSON Schema for models without native structured output.
   - Model capability checks query `models.dev` (via `LlmFactory.info.profile`) for PDF/vision input modalities.
5. **Neutral Documentation & Glossaries**:
   `LinkMLContract` extracts `SchemaDocumentation` and formats compact enum glossaries (`get_enum_glossary()`) for prompts so permissible-value descriptions are never lost.
6. **Manifest Caching**:
   `ManifestCache` checks source file content hash, schema/import closure hash, prompt template hash, model ID, and pipeline version.

## Commands

```bash
# Run extraction on a single input
uv run cli linkml run resume -i "John Smith; SW engineer"

# Batch extract from files
uv run cli linkml extract ./docs ./output --config resume

# Code generation & drift check
uv run cli linkml generate schema.yaml --out generated.py
uv run cli linkml validate-drift schema.yaml generated.py
just linkml-validate-drift schema.yaml generated.py

# Inspect schema documentation and enum glossary
uv run cli linkml show-docs schema.yaml
uv run cli linkml show-glossary schema.yaml

# Run tests
uv run pytest tests/unit_tests/extra/test_linkml_contract.py tests/unit_tests/extra/test_pydantic_processor.py
```

## Avoid

- Never import `genai-graph` inside `genai-tk`.
- Do not generate Python dynamically at runtime; use build-time generation.
- Do not let LLM hallucinate enums without providing the LinkML enum glossary in the prompt.
- Do not assume all providers accept raw PDF files; rely on `models.dev` capability checks.
