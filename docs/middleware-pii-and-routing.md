# PII Anonymization & Sensitivity Routing Middleware

This document describes two LangChain middlewares for securing agent conversations:

1. **AnonymizationMiddleware** — Detects and redacts PII before reaching the LLM, then restores it in responses
2. **SensitivityRouterMiddleware** — Routes sensitive conversations to a safer LLM based on content and RAG sources

Both middlewares are thread-isolated (concurrent conversations remain separate) and composable.

## AnonymizationMiddleware

### Overview

Reversible PII anonymization using [Presidio](https://microsoft.github.io/presidio/) for entity detection and [Faker](https://faker.readthedocs.io/) for generating fake replacements.

- **Before model**: Detects PII in user messages, replaces with Faker-generated values, stores mapping
- **After model**: Restores original PII in LLM responses using the stored mapping
- Supports fuzzy matching for de-anonymization (handles minor text variations)

### Configuration

```python
from genai_tk.agents.langchain.middleware.anonymization_middleware import (
    AnonymizationMiddleware,
    AnonymizationConfig,
)
from genai_tk.agents.langchain.middleware.presidio_detector import PresidioDetectorConfig

config = AnonymizationConfig(
    detector=PresidioDetectorConfig(
        analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
        enable_spacy=True,
        spacy_model="en_core_web_sm",
    ),
    faker_seed=42,  # reproducible fake values
    faker_locales=["en_US"],
    fuzzy_deanonymize=True,
    fuzzy_threshold=85,  # 0-100; higher = stricter matching
)

middleware = AnonymizationMiddleware(config=config)
```

### YAML Configuration

Add to `config/agents/langchain.yaml`:

```yaml
langchain_agents:
  profiles:
    - name: PrivacyAgent
      type: react
      llm: default
      middlewares:
        - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
          analyzed_fields:
            - PERSON
            - EMAIL_ADDRESS
            - PHONE_NUMBER
            - CREDIT_CARD
          faker_seed: 42
          fuzzy_deanonymize: true
          fuzzy_threshold: 85
```

Then run:
```bash
uv run cli agents langchain --profile PrivacyAgent --chat
```

### Supported Entity Types

Presidio's built-in recognizers include:
- `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`
- `CREDIT_CARD`, `IBAN_CODE`, `IMEI_ID`
- `IP_ADDRESS`, `URL`, `DOMAIN_NAME`
- `US_SOCIAL_SECURITY_NUMBER`, `LOCATION`, etc.

See [Presidio docs](https://microsoft.github.io/presidio/supported_entities/) for the complete list.

### Custom Recognizers

Add domain-specific entity types using regex patterns and optional context words via `CustomRecognizerConfig`:

```python
from genai_tk.agents.langchain.middleware.presidio_detector import (
    CustomRecognizerConfig,
    PresidioDetectorConfig,
)

config = PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS"],
    custom_recognizers=[
        # Match internal project codes: PRJ-12345
        CustomRecognizerConfig(
            entity_name="PROJECT_CODE",
            patterns=[r"\bPRJ-\d{5}\b"],
            context=["project", "ticket", "issue"],
            score=0.85,
        ),
        # Match internal employee IDs: EMP-A1234
        CustomRecognizerConfig(
            entity_name="EMPLOYEE_ID",
            patterns=[r"\bEMP-[A-Z]\d{4}\b"],
            context=["employee", "staff", "hr"],
            score=0.9,
        ),
    ],
)
```

Custom entity types are anonymized with a generic `ENTY####` placeholder by default. To provide a better Faker replacement, subclass `AnonymizationMiddleware` and override `_fake_value()`:

```python
class MyMiddleware(AnonymizationMiddleware):
    def _fake_value(self, entity_type: str) -> str:
        if entity_type == "PROJECT_CODE":
            return f"PRJ-{self._faker.numerify('#####')}"
        if entity_type == "EMPLOYEE_ID":
            return f"EMP-{self._faker.bothify('?####').upper()}"
        return super()._fake_value(entity_type)
```

#### YAML Custom Recognizers

```yaml
langchain_agents:
  profiles:
    - name: PrivacyAgent
      type: react
      llm: default
      middlewares:
        - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
          analyzed_fields: [PERSON, EMAIL_ADDRESS, PROJECT_CODE]
          custom_recognizers:
            - entity_name: PROJECT_CODE
              patterns:
                - '\bPRJ-\d{5}\b'
              context: [project, ticket]
              score: 0.85
```

### Thread Isolation

The anonymization mapping is keyed by `thread_id`, so each concurrent conversation maintains separate PII mappings:

```python
agent.invoke(
    {"messages": [HumanMessage(content="My email is alice@example.com")]},
    config={"configurable": {"thread_id": "conversation-1"}},
)
# → PII mapped under conversation-1

agent.invoke(
    {"messages": [HumanMessage(content="My email is bob@example.com")]},
    config={"configurable": {"thread_id": "conversation-2"}},
)
# → PII mapped under conversation-2 (separate from conversation-1)
```

### Cleanup

To remove mappings for a completed conversation:

```python
middleware.cleanup("conversation-1")
```

## SensitivityRouterMiddleware

### Overview

Routes agent messages to a "safe" LLM when content is detected as sensitive. Sensitivity is determined by:

1. **Content analysis** — Regex patterns (API keys, private keys, etc.) + keyword groups (credentials, financial, medical)
2. **RAG source patterns** — Documents matching glob patterns (e.g., `**/hr/**`, `**/executive/**`)

When either signal indicates sensitivity, the LLM is switched to a pre-configured safer model.

### Configuration

```python
from genai_tk.agents.langchain.middleware.sensitivity_router_middleware import (
    SensitivityRouterMiddleware,
    RouterConfig,
)
from genai_tk.agents.langchain.middleware.sensitivity_scorer import SensitivityScorerConfig

config = RouterConfig(
    safe_llm="gpt_4mini@openai",  # fallback for sensitive content
    sensitive_file_patterns=[
        "**/hr/**",
        "**/executive/**",
        "**/medical/**",
        "**/*confidential*",
    ],
    scorer_config=SensitivityScorerConfig(
        threshold=0.5,  # 0.0-1.0; higher = less sensitive
    ),
)

middleware = SensitivityRouterMiddleware(config=config)
```

### YAML Configuration

```yaml
langchain_agents:
  profiles:
    - name: SecurityAgent
      type: react
      llm: gpt_41@openai  # default LLM
      middlewares:
        - class: genai_tk.agents.langchain.middleware.sensitivity_router_middleware:SensitivityRouterMiddleware
          safe_llm: gpt_4mini@openai
          sensitive_file_patterns:
            - "**/hr/**"
            - "**/executive/**"
            - "**/medical/**"
          threshold: 0.5
```

### Sensitivity Scoring

The `SensitivityScorer` detects:

**Regex Patterns** (default):
- Email addresses, API keys, private keys, credit cards, IP addresses, URLs

**Keyword Groups** (default):
- `credentials`: password, secret, token, auth, key
- `financial`: salary, budget, revenue, price, cost, payment
- `medical`: patient, doctor, diagnosis, prescription, health
- `personal`: ssn, dob, address, phone, id

**Banwords** (hardcoded):
- `root`, `admin`, `secret`, `confidential`

Custom patterns can be provided via `SensitivityScorerConfig`:

```python
from genai_tk.agents.langchain.middleware.sensitivity_scorer import SensitivityScorerConfig

config = SensitivityScorerConfig(
    threshold=0.6,
    regex_patterns={
        "custom_api": r"(api_[a-z0-9]{32})",  # Custom API key format
    },
    keyword_groups={
        "custom_domain": ["proprietary", "classified"],
    },
)
```

### Thread Isolation

Like `AnonymizationMiddleware`, each thread maintains its own sensitivity state. Once a thread is marked as sensitive (either by content or RAG source), all subsequent model calls use the safe LLM.

```python
# Thread A retrieves a sensitive document
agent.invoke(
    {"messages": [HumanMessage(content="What's in the HR file?")]},
    config={"configurable": {"thread_id": "thread-a"}},
)
# → Document source matches **/hr/** → thread-a marked as sensitive

# Follow-up in thread A uses safe LLM
agent.invoke(
    {"messages": [HumanMessage(content="And the salary range?")]},
    config={"configurable": {"thread_id": "thread-a"}},
)
# → Uses safe LLM even though content isn't flagged (sticky routing)

# Thread B is unaffected
agent.invoke(
    {"messages": [HumanMessage(content="Tell me about the product")]},
    config={"configurable": {"thread_id": "thread-b"}},
)
# → Uses default LLM
```

## Usage Examples

### Example 1: Privacy-Preserving Agent

```python
from genai_tk.agents.langchain.langchain_agent import LangchainAgent

agent = LangchainAgent(
    profile_name="PrivacyAgent",  # from langchain.yaml
)

result = agent.run("I'm Alice Johnson (alice.j@company.com). Can you look up my record?")
# → LLM sees anonymized name/email
# → Response is de-anonymized before returning to user
```

### Example 2: Secure RAG with Sensitivity Routing

```python
agent = LangchainAgent(
    profile_name="SecurityAgent",
)

# Thread 1: Public FAQ
result = agent.run("What's the product pricing?")  # Uses default fast LLM

# Thread 2: HR data
result = agent.run(
    "What's the salary band for engineers?",
    config={"configurable": {"thread_id": "hr-query"}},
)
# → Tool retrieves from docs/hr/bands.pdf
# → Source matches **/hr/** → routes to safe LLM
# → All follow-ups in this thread use safe LLM
```

### Example 3: Combined Anonymization + Routing

```yaml
langchain_agents:
  profiles:
    - name: SecurePrivacyAgent
      type: react
      llm: gpt_41@openai
      middlewares:
        # Innermost: Anonymize PII
        - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
          analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER]
          faker_seed: 42
        
        # Outermost: Route sensitive content
        - class: genai_tk.agents.langchain.middleware.sensitivity_router_middleware:SensitivityRouterMiddleware
          safe_llm: gpt_4mini@openai
          sensitive_file_patterns: ["**/executive/**"]
          threshold: 0.5
```

## Implementation Details

### Thread ID Extraction

Both middlewares use `langgraph.config.get_config()` to retrieve the current `thread_id` from the LangGraph execution context. This works seamlessly in:

- Real agent runs via `invoke()` / `ainvoke()` with `config={"configurable": {"thread_id": ...}}`
- Unit tests via mock `runtime.config` / `request.config` attributes

### Per-Thread State

- `AnonymizationMiddleware._mapping: dict[str, dict[str, str]]` — Maps `thread_id → {original_pii → fake_value}`
- `SensitivityRouterMiddleware._sensitive_sources: dict[str, set[str]]` — Maps `thread_id → {source_files}`

### Composability

Middleware are applied as a stack (first in list = outermost):

```python
middlewares = [outer_mw, inner_mw]
# Execution order: outer_mw.before_model → inner_mw.before_model → LLM → inner_mw.after_model → outer_mw.after_model
```

## Testing

Unit tests are located in:
- `tests/unit_tests/agents/langchain/middleware/test_anonymization_middleware.py` (19 tests)
- `tests/unit_tests/agents/langchain/middleware/test_sensitivity_router_middleware.py` (20 tests)
- `tests/unit_tests/agents/langchain/middleware/test_sensitivity_scorer.py` (24 tests)

Run all middleware tests:
```bash
uv run pytest tests/unit_tests/agents/langchain/middleware/ -v
```

## Demos

Interactive notebooks demonstrating both middlewares:
- `notebooks/middleware_anonymization_demo.ipynb` — Low-level anonymization + agent demo
- `notebooks/middleware_router_demo.ipynb` — Content/RAG-based routing with tool integration

## See Also

- [LangChain Middleware Guide](https://python.langchain.com/docs/concepts/agents#middleware)
- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [Faker Documentation](https://faker.readthedocs.io/)
