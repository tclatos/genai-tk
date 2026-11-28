# Structured Output with LinkML & Pydantic

[LinkML](https://linkml.io/) (Linked Open Data Modeling Language) is the authoritative schema language
in `genai-tk` for defining structured domain models, extraction targets, permissible values, and graph mappings.

GenAI Toolkit integrates LinkML for **structured extraction** — executing an LLM prompt that returns
a typed, validated Pydantic v2 model instance with schema recovery and manifest-based caching.

---

## Overview: CLI Commands & Workflows

The `cli linkml` command group provides end-to-end tooling for schema inspection, code generation,
CI drift detection, and workflow execution.

| CLI Command | Workflow | Purpose |
|---|---|---|
| `cli linkml run` | `linkml_run` | Extract structured data from a single text input or file |
| `cli linkml extract` | `linkml_extract` | Batch-process a directory of Markdown/PDF files |
| `cli linkml generate` | — | Generate Pydantic v2 models at build time |
| `cli linkml validate-drift` | — | CI drift check between LinkML schema and generated Python |
| `cli linkml show-docs` | — | Display neutral schema documentation (classes, fields, enums) |
| `cli linkml show-glossary` | — | Display formatted Markdown glossary of enum permissible values |

---

## Quickstart

### 1. Install LinkML Extra

```bash
uv sync --extra linkml
# Or in consuming projects:
# uv add "genai-tk[linkml]"
```

### 2. Define a LinkML Schema

```yaml
# schema/resume.yaml
id: https://example.org/tests/resume
name: resume
imports:
  - linkml:types
default_range: string

classes:
  Resume:
    tree_root: true
    description: Candidate resume containing contact and background details.
    attributes:
      name:
        range: string
        required: true
        description: Full name of the candidate
      email:
        range: string
        required: true
        description: Email address
      experience:
        range: string
        multivalued: true
        description: Professional experiences
      skills:
        range: string
        multivalued: true
        description: Technical skills
```

### 3. Generate Pydantic Models & Check Drift

```bash
# Generate models
uv run cli linkml generate schema/resume.yaml --out src/generated/resume.py

# Validate drift in CI
uv run cli linkml validate-drift schema/resume.yaml src/generated/resume.py
# Or with just:
just linkml-validate-drift schema/resume.yaml src/generated/resume.py
```

### 4. Configure in `app_conf.yaml`

```yaml
linkml:
  resume:
    schema: schema/resume.yaml
    root_class: Resume
    generated_module: src.generated.resume
    prompt: prompts/resume.md
    llm: default
```

### 5. Run Extraction

```bash
# Extract from command line input
uv run cli linkml run resume -i "John Smith; Senior Python Engineer; john@example.com"

# Extract from file and save result
uv run cli linkml run resume --file candidate.md --out-dir ./output --out-file john.json

# Batch process directory
uv run cli linkml extract ./resumes ./output --config resume --pathspec '**/*.md'
```

---

## Robust Extraction Architecture

`LinkMLProcessor` implements a three-tier execution strategy:

```text
Document / Prompt
       │
       ▼
LangChain Chat Model (with_structured_output(include_raw=True))
       │
       ├─► Fast Path: parsed is valid Pydantic model ──► Return model
       │
       ├─► Recovery Path: json_repair in schema mode on raw output
       │     └─► Repaired JSON matches model_json_schema() ──► Return validated model
       │
       └─► Fallback Path: JSON Schema prompt appended for models lacking native structured output
```

### Model Capabilities via `models.dev`

Before invoking multimodal or document features, the processor validates the model's capabilities
against the local `models.dev` database (`models_dev.json`):
- If a document is a PDF and the model does not declare PDF modality, a clear [ModelCapabilityError](../genai_tk/extra/structured/exceptions.py) is raised.
- If vision input is required and unsupported, execution halts early with actionable guidance.

---

## Python API

```python
from genai_tk.extra.structured.linkml_contract import LinkMLContract
from genai_tk.extra.structured.linkml_processor import LinkMLProcessor
from my_project.generated.resume import Resume

# 1. Load contract
contract = LinkMLContract(
    schema_path="schema/resume.yaml",
    root_class="Resume",
    generated_module="my_project.generated.resume",
)

# 2. Initialize processor
processor = LinkMLProcessor(
    model_cls=Resume,
    prompt="prompts/resume.md",
    llm="default",
    contract=contract,
)

# 3. Extract
resume = processor.extract("doc_1", "John Smith; SW engineer; john@example.com")
print(resume.name, resume.skills)
```
