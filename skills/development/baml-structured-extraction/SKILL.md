---
name: baml-structured-extraction
description: Work on BAML structured extraction, BAML CLI commands, processors, utilities, and Prefect BAML workflow integration in genai-tk.
---

# GenAI Toolkit BAML Structured Extraction

## Read First

- `docs/baml.md`
- `genai_tk/extra/structured/baml_processor.py`
- `genai_tk/extra/structured/baml_util.py`
- `genai_tk/extra/structured/commands_baml.py`
- `genai_tk/workflow/prefect/flows/baml_flow.py`

## Code Map

| Concern | Paths |
|---|---|
| Processor | `genai_tk/extra/structured/baml_processor.py` |
| Utilities | `genai_tk/extra/structured/baml_util.py` |
| CLI | `genai_tk/extra/structured/commands_baml.py` |
| Workflow flow | `genai_tk/workflow/prefect/flows/baml_flow.py` |
| Tests | `tests/unit_tests/extra/test_baml_prefect_flow.py`, `tests/integration_tests/test_commands_baml.py` |

## Downstream Consumers

`genai-graph` is the main consumer of `BamlStructuredProcessor` outside genai-tk: its
`MarkdownBamlFactory` (`genai_graph/kg/factories/markdown_baml_factory.py`) calls
`processor.analyze_document(...)` inline to extract structured entities from Markdown and
MERGE them into a Knowledge Graph, caching the result as JSON build artifacts. When
changing the processor's contract or return shape, check that factory and the
`genai-graph/skills/genai-graph/kg-factories` skill still hold — see
`genai-graph/docs/baml_extraction_guide.md` and `genai-graph/docs/document-graph.md`.

## Change Workflow

1. Keep BAML schemas and generated behavior aligned with `docs/baml.md`.
2. Validate structured outputs with Pydantic models at the boundary.
3. Keep CLI behavior and Prefect flow behavior consistent.
4. Add unit tests for processor/flow logic and integration tests for CLI commands.

## Commands

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/extra/test_baml_prefect_flow.py -q
uv run pytest tests/integration_tests/test_commands_baml.py -q
uv run cli baml --help
```

## Avoid

- Do not put provider secrets in BAML examples or generated artifacts.
- Do not let raw unvalidated dicts cross public API boundaries.
- Do not duplicate extraction examples between docs and module docstrings.
