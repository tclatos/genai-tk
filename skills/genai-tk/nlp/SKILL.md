---
name: genai-tk-nlp
description: Configure and extend spaCy-based NLP in genai-tk — covers the centralized NLP package (genai_tk.extra.nlp), the nlp: YAML config section, spaCy model management, text preprocessing for BM25, PII detection with Presidio, anonymization, French and multilingual support, the TextClassifier protocol, and the DefaultSensitivityScorer. Use when adding NLP, language detection, preprocessing, PII detection, or text classification to any part of the toolkit.
tags: [nlp, spacy, presidio, pii, anonymization, bm25, french, classifier]
version: "1.0"
---

# NLP in genai-tk (`genai_tk.extra.nlp`)

## Read First

- `docs/nlp.md` — complete reference
- `genai_tk/extra/nlp/` — all NLP code
- `genai_tk/extra/nlp/config.py` — `NlpConfig`, `nlp_config()`
- `genai_tk/extra/nlp/engine.py` — `get_nlp()`
- `genai_tk/extra/nlp/model_manager.py` — `SpaCyModelManager`
- `genai_tk/extra/nlp/preprocessing.py` — `get_spacy_preprocess_fn()`
- `genai_tk/extra/nlp/presidio.py` — `PresidioDetector`, `PresidioDetectorConfig`
- `genai_tk/extra/nlp/anonymization.py` — `anonymize_text()`, `AnonymizationConfig`
- `genai_tk/extra/nlp/classifiers/` — `TextClassifier`, `DefaultSensitivityScorer`

## Key Principle

**All NLP code goes through `genai_tk.extra.nlp`.** Do not call `spacy.load()` directly, do not import from old paths like `genai_tk.workflow.anonymization` or `genai_tk.utils.spacy_model_mngr` in new code.

## Module Map

```
genai_tk/extra/nlp/
  __init__.py          ← public API (all symbols below re-exported here)
  config.py            ← NlpConfig, nlp_config()
  engine.py            ← get_nlp(language?, model?) → spacy.Language
  model_manager.py     ← SpaCyModelManager (download, setup, require)
  preprocessing.py     ← get_spacy_preprocess_fn(), default_preprocessing_func()
  presidio.py          ← PresidioDetector, PresidioDetectorConfig, CustomRecognizerConfig, DetectedEntity
  anonymization.py     ← anonymize_text(), AnonymizationConfig, make_fake_value()
  classifiers/
    __init__.py
    base.py            ← TextClassifier (protocol), ClassificationResult
    sensitivity.py     ← DefaultSensitivityScorer, DefaultScorerConfig, SensitivityAssessment
```

## Installation

```bash
uv sync --extra nlp
```

Installs: `spacy`, `en_core_web_sm`, `en_core_web_lg`, `presidio-analyzer`, `presidio-anonymizer`.

## Configuration

```yaml
# config/app_conf.yaml
nlp:
  default_language: en
  default_model: en_core_web_sm
  models:
    en: en_core_web_sm
    fr: fr_core_news_sm
```

Access from Python:
```python
from genai_tk.extra.nlp import nlp_config
cfg = nlp_config()
model = cfg.get_model_for_language("fr")  # "fr_core_news_sm"
```

## Loading spaCy

```python
from genai_tk.extra.nlp import get_nlp

nlp = get_nlp()                    # NlpConfig defaults
nlp_fr = get_nlp(language="fr")    # French model
nlp_lg = get_nlp(model="en_core_web_lg")  # explicit override
```

Always use `get_nlp()` — it checks the feature gate, resolves config, and caches.

## Text Preprocessing (BM25)

```python
from genai_tk.extra.nlp import get_spacy_preprocess_fn

preprocess = get_spacy_preprocess_fn()        # NlpConfig defaults
preprocess_fr = get_spacy_preprocess_fn(language="fr")
tokens = preprocess("The quick brown foxes jumping")
# → ["quick", "brown", "fox", "jump"]
```

BM25 retriever config:
```yaml
retriever:
  type: bm25
  preprocessing: spacy    # uses NlpConfig when spacy_model is not specified
```

## PII Detection

```python
from genai_tk.extra.nlp import PresidioDetector, PresidioDetectorConfig, DetectedEntity

detector = PresidioDetector(config=PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS"],
    language="fr",      # None → NlpConfig.default_language
    spacy_model=None,   # None → NlpConfig.models[language]
    score_threshold=0.4,
))
entities: list[DetectedEntity] = detector.detect("Contact Jean at jean@acme.fr")
```

## Anonymization

```python
from faker import Faker
from genai_tk.extra.nlp import AnonymizationConfig, PresidioDetector, anonymize_text

config = AnonymizationConfig(faker_seed=42, faker_locales=["en_US"])
detector = PresidioDetector(config=config.detector)
Faker.seed(42)
faker = Faker(["en_US"])

anonymized, mapping = anonymize_text("Alice at alice@example.com", detector=detector, faker=faker)
```

## Text Classification

```python
from genai_tk.extra.nlp.classifiers import DefaultSensitivityScorer, DefaultScorerConfig

scorer = DefaultSensitivityScorer()
result = scorer.classify("My API key is sk-abc123...")   # TextClassifier protocol
result = scorer.assess("My API key is sk-abc123...")    # legacy alias

print(result.score)        # float 0-1
print(result.level)        # "low" | "medium" | "high" | "critical"
print(result.is_sensitive) # bool
print(result.labels)       # list of active signal categories
```

Custom classifier implementing the protocol:
```python
from genai_tk.extra.nlp.classifiers.base import ClassificationResult, TextClassifier

class MyClassifier:
    def classify(self, text: str) -> ClassificationResult:
        return ClassificationResult(score=0.5, level="medium", labels=["my_signal"])
```

## French / Multilingual

1. Add to `config/app_conf.yaml`:
   ```yaml
   nlp:
     models:
       fr: fr_core_news_sm
   ```
2. Install model: `python -m spacy download fr_core_news_sm`
3. If missing, `get_nlp(language="fr")` raises:
   ```
   ImportError: spaCy model 'fr_core_news_sm' for language 'fr' is not installed.
   Install it with: python -m spacy download fr_core_news_sm
   ```

## Feature Gate

```python
from genai_tk.config_mgmt.features import is_available, require_feature

if is_available("nlp"):
    from genai_tk.extra.nlp import get_nlp
    nlp = get_nlp()

# At function entry (raises with install instructions if missing):
require_feature("nlp", context="my_function")
```

## Tests

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/extra/ tests/unit_tests/utils/test_spacy.py -q
```

## Avoid

- Do NOT call `spacy.load()` directly — use `get_nlp()`.
- Do NOT import from deprecated paths (`workflow.anonymization.*`, `utils.spacy_model_mngr`) in new code.
- Do NOT import `presidio_analyzer` or `spacy` at module level — use `TYPE_CHECKING` or lazy imports to keep the `nlp` extra truly optional.
