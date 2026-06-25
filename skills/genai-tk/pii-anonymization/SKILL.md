---
name: genai-tk-pii-anonymization
description: Use or extend PII anonymization in genai-tk — covers the shared core library (genai_tk.extra.nlp), the AnonymizationMiddleware for agents, the anonymize_files_flow Prefect ETL flow, custom entity recognizers (COMPANY, PRODUCT, PROJECT), and the SensitivityRouterMiddleware. Use when adding privacy controls to a workflow, an agent profile, or a batch pipeline.
---

# PII Anonymization in genai-tk

## Read First

- `docs/nlp.md` — full NLP reference (spaCy, Presidio, classifiers, French support)
- `docs/middleware-pii-and-routing.md` — agent middleware reference
- `genai_tk/extra/nlp/presidio.py` — `PresidioDetector`, `PresidioDetectorConfig`, `CustomRecognizerConfig`
- `genai_tk/extra/nlp/anonymization.py` — `anonymize_text()`, `make_fake_value()`, `AnonymizationConfig`
- `genai_tk/agents/langchain/middleware/anonymization_middleware.py` — LangChain agent middleware
- `genai_tk/workflow/prefect/flows/anonymize_flow.py` — Prefect ETL flow

## Architecture

```
genai_tk/extra/nlp/
  presidio.py      ← PresidioDetector, PresidioDetectorConfig, CustomRecognizerConfig, DetectedEntity
  anonymization.py ← anonymize_text(), make_fake_value(), AnonymizationConfig

Consumers:
  agents/langchain/middleware/anonymization_middleware.py  ← AnonymizationMiddleware (agent)
  workflow/prefect/flows/anonymize_flow.py                 ← anonymize_files_flow (ETL batch)
```

Both consumers call the **same** `anonymize_text()` function — identical behaviour at ETL time and agent runtime.

## Key Types

| Type | Module | Purpose |
|------|--------|---------|
| `PresidioDetectorConfig` | `extra.nlp.presidio` | Which fields to detect, custom recognizers, spaCy model, score threshold |
| `CustomRecognizerConfig` | `extra.nlp.presidio` | One regex-based custom entity (entity name + patterns + context words) |
| `PresidioDetector` | `extra.nlp.presidio` | Runs Presidio analysis; `AnalyzerEngine` is cached per-config (singleton) |
| `AnonymizationConfig` | `extra.nlp.anonymization` | Wraps detector config + Faker settings + fuzzy-deano config |
| `anonymize_text()` | `extra.nlp.anonymization` | Stateless: detect → fake → return `(anonymized_text, mapping)` |
| `make_fake_value()` | `extra.nlp.anonymization` | Maps entity type → Faker call |
| `AnonymizationMiddleware` | `agents.langchain.middleware.anonymization_middleware` | LangChain `AgentMiddlware`; before_model anonymizes, after_model deanonymizes; thread-safe |
| `SensitivityRouterMiddleware` | `agents.langchain.middleware.sensitivity_router_middleware` | Routes sensitive calls to a safe LLM |

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
      save_mapping: true
```

```bash
uv run cli workflow run anonymize_docs --dry-run
uv run cli workflow run anonymize_docs
```

## Use Case 3 — Programmatic / Scripting

```python
from faker import Faker
from genai_tk.extra.nlp import AnonymizationConfig, CustomRecognizerConfig, PresidioDetector, PresidioDetectorConfig, anonymize_text

config = PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
    custom_recognizers=[
        CustomRecognizerConfig(
            entity_name="COMPANY",
            patterns=[r"(?i)\b(Acme Corp|Tech Solutions)\b"],
            context=["company", "firm"],
        ),
    ],
    score_threshold=0.4,
)
detector = PresidioDetector(config=config)
Faker.seed(42)
faker = Faker(["en_US"])

anonymized, mapping = anonymize_text(
    "John Smith at Acme Corp: john@acme.com",
    detector=detector,
    faker=faker,
)
# Reuse mapping across chunks of the same document
anonymized2, mapping = anonymize_text(chunk2, detector=detector, faker=faker, mapping=mapping)
```

## French / Multilingual PII

Set language in config or per-detector:

```yaml
# config/app_conf.yaml
nlp:
  default_language: fr
  default_model: fr_core_news_sm
  models:
    en: en_core_web_sm
    fr: fr_core_news_sm
```

```python
from genai_tk.extra.nlp import PresidioDetector, PresidioDetectorConfig

detector = PresidioDetector(config=PresidioDetectorConfig(language="fr"))
entities = detector.detect("Appelez Jean Dupont au 06 12 34 56 78")
```

Install French model: `python -m spacy download fr_core_news_sm`

If the model is not installed, you get a clear `ImportError` with install instructions.

## Supported Entity Types

**Presidio built-ins:** `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `LOCATION`, `IBAN_CODE`, `US_SSN`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `NRP`, `ORG`

**Domain-specific (via `CustomRecognizerConfig`):** `COMPANY` → `faker.company()`, `PRODUCT` → `faker.bs()`, `PROJECT` → `faker.bs()`

**Custom fallback:** any unknown entity type → `faker.bothify(text="ABCD####")`

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

## Tests

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/extra/test_anonymization_core.py -q
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/agents/langchain/middleware/ -q
```

## Avoid

- Do NOT store the anonymization mapping without encryption — it maps fake values back to real PII.
