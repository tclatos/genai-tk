# NLP (`genai_tk.extra.nlp`)

> **Quick nav:** [Config](#configuration) · [spaCy Engine](#spacy-engine) · [Preprocessing](#text-preprocessing) · [PII Detection](#pii-detection-presidio) · [Anonymization](#anonymization) · [Classifiers](#text-classifiers) · [French & multilingual](#french--multilingual-support)

## Overview

The `genai_tk.extra.nlp` package is the single home for all NLP and spaCy-related functionality in the toolkit. It consolidates code that was previously scattered across:

| Old location | Now in |
|---|---|
| `genai_tk.workflow.anonymization.presidio_detector` | `genai_tk.extra.nlp.presidio` |
| `genai_tk.workflow.anonymization.core` | `genai_tk.extra.nlp.anonymization` |
| `genai_tk.utils.spacy_model_mngr` | `genai_tk.extra.nlp.model_manager` |
| `genai_tk.workflow.retrievers.bm25s_retriever.get_spacy_preprocess_fn` | `genai_tk.extra.nlp.preprocessing` |
| `genai_tk.agents.langchain.middleware.sensitivity_scorer` (implementation) | `genai_tk.extra.nlp.classifiers.sensitivity` |

Old import paths remain functional with a `DeprecationWarning`. Update imports in new code to use `genai_tk.extra.nlp`.

### Installation

NLP features require the optional `nlp` extra:

```bash
uv sync --extra nlp
# or
uv add "genai-tk[nlp]"
```

This installs spaCy, English models (`en_core_web_sm`, `en_core_web_lg`), and Presidio.

---

## Configuration

### YAML section (`nlp:`)

Add to `config/app_conf.yaml` (or any merged YAML file):

```yaml
nlp:
  default_language: en
  default_model: en_core_web_sm
  models:
    en: en_core_web_sm
    fr: fr_core_news_sm      # requires: python -m spacy download fr_core_news_sm
    de: de_core_news_sm      # requires: python -m spacy download de_core_news_sm
```

All NLP consumers (Presidio, BM25, preprocessing) fall back to these defaults when no explicit model/language is passed.

### Pydantic model & accessor

```python
from genai_tk.extra.nlp.config import NlpConfig, nlp_config

cfg: NlpConfig = nlp_config()
print(cfg.default_language)          # "en"
print(cfg.get_model_for_language("fr"))  # "fr_core_news_sm"
```

`nlp_config()` returns a validated `NlpConfig` with defaults when the `nlp:` section is absent — it never raises.

---

## spaCy Engine

`genai_tk.extra.nlp.engine` is the single entry point for loading spaCy pipelines. All code should call `get_nlp()` instead of `spacy.load()` directly.

```python
from genai_tk.extra.nlp import get_nlp

nlp = get_nlp()                        # uses NlpConfig defaults
nlp_fr = get_nlp(language="fr")        # French model from config
nlp_lg = get_nlp(model="en_core_web_lg")  # explicit model override
```

**Features:**
- Calls `require_feature("nlp")` — raises `ImportError` with install instructions if spaCy is not installed
- Caches loaded pipelines per `(language, model)` pair — loading is expensive; this ensures it happens once
- Returns `spacy.language.Language` (fully typed)
- Clears the cache in tests with `from genai_tk.extra.nlp.engine import clear_cache`

---

## SpaCy Model Manager

`SpaCyModelManager` handles model discovery, download, and installation:

```python
from genai_tk.extra.nlp.model_manager import SpaCyModelManager

# Check availability (global install or custom path)
if not SpaCyModelManager.is_model_installed("fr_core_news_sm"):
    SpaCyModelManager.download_model("fr_core_news_sm")

# Set up model — downloads if needed, verifies load
SpaCyModelManager.setup_spacy_model("en_core_web_sm")

# Raise clear error if model is missing (no auto-download)
SpaCyModelManager.require_model("fr_core_news_sm", language="fr")
```

Models are stored in `<paths.models>/spacy_models/<model_name>`.

---

## Text Preprocessing

`genai_tk.extra.nlp.preprocessing` provides lemmatization + stop-word removal suitable for BM25 and hybrid search:

```python
from genai_tk.extra.nlp.preprocessing import get_spacy_preprocess_fn, default_preprocessing_func

# Whitespace split (no spaCy required)
tokens = default_preprocessing_func("The quick brown fox")
# → ["The", "quick", "brown", "fox"]

# Lemmatization + stop-word removal (requires nlp extra)
preprocess = get_spacy_preprocess_fn()                          # NlpConfig defaults
preprocess_fr = get_spacy_preprocess_fn(language="fr")         # French
preprocess_custom = get_spacy_preprocess_fn(
    model="en_core_web_lg",
    more_stop_words=["foo", "bar"],
)

tokens = preprocess("The quick brown foxes jumping over lazy dogs")
# → ["quick", "brown", "fox", "jump", "lazy", "dog"]
```

This function is used by the BM25 retriever when `preprocessing: spacy` is set.

---

## PII Detection (Presidio)

`genai_tk.extra.nlp.presidio` wraps Microsoft Presidio with optional spaCy NER:

### Basic usage

```python
from genai_tk.extra.nlp import PresidioDetector, PresidioDetectorConfig, DetectedEntity

# Default config — English, en_core_web_sm
detector = PresidioDetector()
entities: list[DetectedEntity] = detector.detect("Call John at john@acme.com")
# → [DetectedEntity(entity_type="PERSON", ...), DetectedEntity(entity_type="EMAIL_ADDRESS", ...)]
```

### PresidioDetectorConfig

```python
config = PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
    language="fr",           # None → NlpConfig.default_language
    spacy_model=None,        # None → NlpConfig.models[language]
    enable_spacy=True,       # False → pattern-only mode (no NER, faster)
    score_threshold=0.4,
    custom_recognizers=[
        CustomRecognizerConfig(
            entity_name="COMPANY",
            patterns=[r"(?i)\b(Acme|Globex|Initech)\b"],
            context=["company", "firm", "corporation"],
        )
    ],
)
detector = PresidioDetector(config=config)
```

### Custom recognizers

```python
from genai_tk.extra.nlp import CustomRecognizerConfig

CustomRecognizerConfig(
    entity_name="PROJECT",
    patterns=[r"(?i)\bProject\s+[A-Z][a-z]+\b"],
    context=["project", "initiative", "programme"],
    score=0.85,
)
```

### Performance

The underlying `AnalyzerEngine` (and spaCy model) is built once per unique config — all `PresidioDetector` instances sharing the same configuration share one cached engine. This is implemented with `@once` singletons.

---

## Anonymization

`genai_tk.extra.nlp.anonymization` provides stateless PII replacement using Faker:

```python
from faker import Faker
from genai_tk.extra.nlp import AnonymizationConfig, PresidioDetector, anonymize_text, make_fake_value

config = AnonymizationConfig(faker_seed=42, faker_locales=["en_US"])
detector = PresidioDetector(config=config.detector)
Faker.seed(config.faker_seed or 0)
faker = Faker(config.faker_locales)

anonymized, mapping = anonymize_text(
    "Contact Alice Smith at alice@example.com",
    detector=detector,
    faker=faker,
)
# anonymized → "Contact [FakeName] at [fake@email.com]"
# mapping    → {"Alice Smith": "FakeName", "alice@example.com": "fake@email.com"}

# Pass mapping across multiple calls to reuse the same replacements
_, mapping = anonymize_text(second_chunk, detector=detector, faker=faker, mapping=mapping)
```

### Entity → Faker mapping

| Entity type | Faker generator |
|---|---|
| `PERSON` | `name()` |
| `EMAIL_ADDRESS` | `email()` |
| `PHONE_NUMBER` | `phone_number()` |
| `CREDIT_CARD` | `credit_card_number()` |
| `LOCATION` | `city()` |
| `IBAN_CODE` | `iban()` |
| `US_SSN` | `ssn()` |
| `IP_ADDRESS` | `ipv4()` |
| `COMPANY`, `ORG` | `company()` |
| `PRODUCT`, `PROJECT` | `bs()` |

Custom entity types fall back to `XXXX####` pattern. Override `make_fake_value()` for custom mappings.

### Agent integration

See [docs/middleware-pii-and-routing.md](middleware-pii-and-routing.md) for the `AnonymizationMiddleware` and `SensitivityRouterMiddleware`.

### ETL / Prefect integration

The `anonymize_files_flow` Prefect flow (`genai_tk.workflow.prefect.flows.anonymize_flow`) calls `anonymize_text()` on documents before they enter the vector store. See [docs/prefect.md](prefect.md).

---

## Text Classifiers

`genai_tk.extra.nlp.classifiers` provides a reusable text classification abstraction:

### TextClassifier protocol

```python
from genai_tk.extra.nlp.classifiers.base import TextClassifier, ClassificationResult

class MyClassifier:
    def classify(self, text: str) -> ClassificationResult:
        score = ...
        return ClassificationResult(
            score=score,
            level="low" if score < 0.25 else "high",
            labels=["custom"],
            details={"signal": score},
        )
```

`ClassificationResult` fields: `score` (0–1), `level` (low/medium/high/critical), `labels`, `details`.

### DefaultSensitivityScorer

The built-in hybrid sensitivity scorer combines five detection strategies:

| Strategy | Cap | Signals |
|---|---|---|
| Regex | 0.55 | email, phone, CC, IBAN, JWT, API keys, bearer tokens, private keys |
| Keywords | 0.18 | credentials, identity, financial, health, confidentiality groups |
| Banwords | 0.50 | "root password", "private key", "prod database dump", etc. |
| Presidio | 0.25 | entity type × confidence score |
| Heuristics | 0.17 | disclosure phrases, digit density, long mixed tokens |

```python
from genai_tk.extra.nlp.classifiers import DefaultSensitivityScorer, DefaultScorerConfig

scorer = DefaultSensitivityScorer()                          # default config
result = scorer.classify("My email is john@example.com")    # TextClassifier protocol
result = scorer.assess("My email is john@example.com")      # legacy alias

print(result.is_sensitive)       # True
print(result.score)              # 0.28
print(result.level)              # "medium"
print(result.labels)             # ["regex", "presidio"]
print(result.detected_entities)  # [DetectedEntity(...)]
```

#### Custom config

```python
scorer = DefaultSensitivityScorer(DefaultScorerConfig(
    sensitivity_threshold=0.50,       # raise the bar
    entity_weights={"CREDIT_CARD": 0.5, "EMAIL_ADDRESS": 0.3},
    banwords=["internal only", "do not distribute"],
))
```

#### Use in middleware

The `SensitivityRouterMiddleware` accepts any `DefaultSensitivityScorer` (or anything with an `assess()` method). See [docs/middleware-pii-and-routing.md](middleware-pii-and-routing.md).

---

## BM25 + spaCy Preprocessing

The BM25 retrievers use spaCy for optional lemmatization-based preprocessing. Configure via the `nlp:` section — no need to specify the model explicitly:

```yaml
# config/retrievers.yaml
retriever:
  type: bm25
  preprocessing: spacy    # "default" = whitespace split; "spacy" = lemmatize + stop-words
  # spacy_model: en_core_web_sm  ← optional override; uses NlpConfig when absent
```

```python
from genai_tk.core.factories.retriever_factory import BM25RetrieverConfig

cfg = BM25RetrieverConfig(preprocessing="spacy")   # spacy_model=None → uses NlpConfig
print(cfg.resolve_spacy_model())                   # "en_core_web_sm"
```

---

## French & Multilingual Support

### Configuration

```yaml
nlp:
  default_language: fr
  default_model: fr_core_news_sm
  models:
    en: en_core_web_sm
    fr: fr_core_news_sm
```

### Install the French model

```bash
python -m spacy download fr_core_news_sm
# or for better NER:
python -m spacy download fr_core_news_lg
```

### Error handling

If a French model is not installed but `language="fr"` is requested, you get a clear error:

```
ImportError: spaCy model 'fr_core_news_sm' for language 'fr' is not installed.
Install it with: python -m spacy download fr_core_news_sm
Or configure a different model in your config YAML under 'nlp.models.fr'.
```

### French PII detection

```python
from genai_tk.extra.nlp import PresidioDetector, PresidioDetectorConfig

detector = PresidioDetector(config=PresidioDetectorConfig(language="fr"))
entities = detector.detect("Appelez Jean Dupont au 06 12 34 56 78")
```

> **Note:** Presidio's French recognizers cover phone numbers, email, and some identifiers. Person name detection relies on the `fr_core_news_sm` NER model.

---

## Feature Gate

All NLP entry points call `require_feature("nlp")` — if spaCy is not installed, a `ImportError` is raised with install instructions instead of a cryptic `ModuleNotFoundError`.

```python
from genai_tk.config_mgmt.features import is_available

if is_available("nlp"):
    from genai_tk.extra.nlp import get_nlp
    nlp = get_nlp()
```

---

## Module Map

```
genai_tk/extra/nlp/
  __init__.py          ← public API re-exports
  config.py            ← NlpConfig, nlp_config()
  engine.py            ← get_nlp(), clear_cache()
  model_manager.py     ← SpaCyModelManager
  preprocessing.py     ← get_spacy_preprocess_fn(), default_preprocessing_func()
  presidio.py          ← PresidioDetector, PresidioDetectorConfig, DetectedEntity, CustomRecognizerConfig
  anonymization.py     ← anonymize_text(), AnonymizationConfig, make_fake_value()
  classifiers/
    __init__.py
    base.py            ← TextClassifier (protocol), ClassificationResult
    sensitivity.py     ← DefaultSensitivityScorer, DefaultScorerConfig, SensitivityAssessment
```
