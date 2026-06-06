---
name: genai-tk-pii-anonymization
description: Use or extend PII anonymization in genai-tk — covers the shared core library, the AnonymizationMiddleware for agents, the anonymize_files_flow Prefect ETL flow, custom entity recognizers (COMPANY, PRODUCT, PROJECT), and the SensitivityRouterMiddleware. Use when adding privacy controls to a workflow, an agent profile, or a batch pipeline.
---

# PII Anonymization in genai-tk

## Read First

- `docs/middleware-pii-and-routing.md` — full reference for agent middleware
- `genai_tk/workflow/anonymization/core.py` — shared `anonymize_text()` + `make_fake_value()`
- `genai_tk/workflow/anonymization/presidio_detector.py` — `PresidioDetector`, `PresidioDetectorConfig`, `CustomRecognizerConfig`
- `genai_tk/agents/langchain/middleware/anonymization_middleware.py` — LangChain agent middleware
- `genai_tk/workflow/prefect/flows/anonymize_flow.py` — Prefect ETL flow

## Architecture

```
genai_tk/workflow/anonymization/
  core.py              ← anonymize_text(), make_fake_value(), AnonymizationConfig
  presidio_detector.py ← PresidioDetector, PresidioDetectorConfig, CustomRecognizerConfig

Consumers:
  agents/langchain/middleware/anonymization_middleware.py  ← AnonymizationMiddleware (agent)
  workflow/prefect/flows/anonymize_flow.py                 ← anonymize_files_flow (ETL batch)
```

Both consumers call the **same** `anonymize_text()` function — identical behaviour at ETL time and agent runtime.

## Key Types

| Type | Module | Purpose |
|------|--------|---------|
| `PresidioDetectorConfig` | `presidio_detector` | Which fields to detect, custom recognizers, spaCy model, score threshold |
| `CustomRecognizerConfig` | `presidio_detector` | One regex-based custom entity (entity name + patterns + context words) |
| `PresidioDetector` | `presidio_detector` | Runs Presidio analysis; `AnalyzerEngine` is cached per-config (singleton) |
| `AnonymizationConfig` | `core` | Wraps detector config + Faker settings + fuzzy-deano config |
| `anonymize_text()` | `core` | Stateless: detect → fake → return `(anonymized_text, mapping)` |
| `make_fake_value()` | `core` | Maps entity type → Faker call |
| `AnonymizationMiddleware` | `anonymization_middleware` | LangChain `AgentMiddleware`; before_model anonymizes, after_model deanonymizes; thread-safe |
| `SensitivityRouterMiddleware` | `sensitivity_router_middleware` | Routes sensitive calls to a safe LLM |

## Use Case 1 — Add Anonymization to an Agent Profile (YAML)

```yaml
# config/agents/langchain/simple.yaml
langchain_agents:
  privacy_agent:
    name: "Privacy Agent"
    type: react
    llm: default
    middlewares:
      - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
        analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD]
        faker_seed: 42
        fuzzy_deanonymize: true
        fuzzy_threshold: 85
```

Run: `cli agents langchain -p privacy_agent --chat`

**With custom domain entities (companies, products):**

```yaml
middlewares:
  - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
    analyzed_fields: [PERSON, EMAIL_ADDRESS]
    faker_seed: 42
    custom_recognizers:
      - entity_name: COMPANY
        patterns:
          - "(?i)\\b(Acme Corp|Globex|Tech Solutions)\\b"
        context: [company, firm, organization]
      - entity_name: PRODUCT
        patterns:
          - "(?i)\\b(WidgetPro|CloudMaster)\\b"
        context: [product, service, platform]
```

## Use Case 2 — Batch Anonymize Files (Prefect ETL)

Configure via workflow YAML:

```yaml
# config/workflows.yaml
workflows:
  anonymize_docs:
    run: genai_tk.workflow.prefect.flows.anonymize_flow.anonymize_files_flow
    inputs:
      base_dir: "${paths.data_root}/raw"
      output_dir: "${paths.data_root}/anonymized"
      pathspecs: ["**/*.txt", "**/*.md"]
      analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER]
      faker_seed: 42
      save_mapping: true   # writes .mapping.json sidecars for later deanonymization
```

Run:
```bash
uv run cli workflow run anonymize_docs --dry-run
uv run cli workflow run anonymize_docs
```

## Use Case 3 — Programmatic / Scripting

```python
from faker import Faker
from genai_tk.workflow.anonymization.core import AnonymizationConfig, anonymize_text
from genai_tk.workflow.anonymization.presidio_detector import (
    CustomRecognizerConfig, PresidioDetector, PresidioDetectorConfig,
)

config = PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
    custom_recognizers=[
        CustomRecognizerConfig(
            entity_name="COMPANY",
            patterns=[r"(?i)\b(Acme Corp|Tech Solutions)\b"],
            context=["company", "firm"],
        ),
    ],
    score_threshold=0.4,  # lower = more sensitive
)
detector = PresidioDetector(config=config)
Faker.seed(42)
faker = Faker(["en_US"])

anonymized, mapping = anonymize_text(
    "John Smith at Acme Corp: john@acme.com",
    detector=detector,
    faker=faker,
)
# mapping: {"John Smith": "Jane Doe", "Acme Corp": "Hoeger LLC", "john@acme.com": "fake@example.net"}

# Reuse mapping across chunks of the same document so the same entity
# always gets the same fake value:
anonymized2, mapping = anonymize_text(chunk2, detector=detector, faker=faker, mapping=mapping)

# Deanonymize (reverse the mapping):
result = anonymized
for real, fake in mapping.items():
    result = result.replace(fake, real)
```

## Supported Entity Types

**Presidio built-ins:**
`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `LOCATION`, `IBAN_CODE`,
`US_SSN`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `NRP`, `ORG`

**Domain-specific (via `CustomRecognizerConfig`):**
`COMPANY` → `faker.company()`, `PRODUCT` → `faker.bs()`, `PROJECT` → `faker.bs()`

**Custom fallback:** any unknown entity type → `faker.bothify(text="ABCD####")`

To add a richer Faker mapping for a new entity type, add it to `make_fake_value()` in
`genai_tk/workflow/anonymization/core.py`.

## Thread Safety (Agent Middleware)

`AnonymizationMiddleware` isolates PII mappings per `thread_id` (from `runtime.config`).
Parallel conversations never share or leak each other's mapping.

Call `middleware.cleanup(thread_id)` when a conversation is complete to free memory.

## Combining with SensitivityRouterMiddleware

```yaml
middlewares:
  - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
    analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER]
    faker_seed: 42
  - class: genai_tk.agents.langchain.middleware.sensitivity_router_middleware:SensitivityRouterMiddleware
    safe_llm: ollama_local
    sensitive_source_patterns: ["**/hr/**", "**/confidential/**"]
```

Middlewares execute in order: PII is stripped before the sensitivity check runs.

## Tests

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/extra/test_anonymization_core.py -q
```

## Avoid

- Do NOT use `CustomizedPresidioAnonymizer` — it has been removed. Use `PresidioDetector` + `anonymize_text()` instead.
- Do NOT share a `PresidioDetector` instance across threads without locking (the internal `AnalyzerEngine` is thread-safe, but the `_analyzer` lazy-init is not protected — the `once()` cache handles it at the config level).
- Do NOT store the anonymization mapping in a database without encryption — it maps fake values back to real PII.
