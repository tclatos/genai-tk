# LLM Selection and Configuration

This document explains how models are declared, selected at runtime, and inspected
with the `cli info` commands. It also covers the `models.dev` database used for
automatic capability resolution.

## Model identifier format

All LLMs are referenced with the format **`model_id@provider`**:

| Part | Meaning | Example |
|------|---------|---------|
| `model_id` | Logical name defined in `llm.yaml` | `gpt41mini` |
| `provider` | Provider key defined in `providers.yaml` | `openai` |
| Combined | Full identifier used everywhere | `gpt41mini@openai` |

You can omit `@provider` when passing a raw provider model name to the CLI
(e.g. `gpt-4o-mini`) — LiteLLM resolves it via fuzzy matching from the
`models.dev` database.

---

## Declaring a model in YAML

Models are declared in `config/providers/llm.yaml`:

```yaml
llm:
  exceptions:

    - model_id: gpt41mini           # logical name (Python-identifier-safe)
      providers:
        - openai: gpt-4.1-mini-2025-04-14   # direct provider + native model name

    - model_id: haiku
      providers:
        - openrouter: anthropic/claude-haiku-4-5   # via gateway

    - model_id: mistral_7b
      providers:
        - custom:                    # full ChatOpenAI constructor params
            model: mistralai/Mistral-7B-Instruct-v0.3
            base_url: https://my-gateway.example.com/v1
            api_key: ${oc.env:MY_GATEWAY_KEY}

    - model_id: parrot_local        # fake model for tests — no API key needed
      providers:
        - fake: parrot
```

`providers:` is a **priority list** — the first provider with a valid API key is used.

---

## Setting the default model

The default model is configured in `config/baseline.yaml`:

```yaml
llm:
  default: gpt41mini@openai    # used when -m / --llm is not specified
```

Override per-session with the `LLM_DEFAULT` environment variable, or always pass
`-m <model>` on the CLI.

---

## Named tags

Tags are aliases for common models, usable with `--llm <tag>`:

```yaml
# config/baseline.yaml
llm:
  tags:
    fast_model:  gpt41mini@openai
    cheap_model: haiku@openrouter
    local:       mistral@ollama
```

```bash
cli core llm -i "Explain RAG" -m fast_model
```

---

## Runtime selection

```bash
# Use default from config
cli core llm -i "Explain RAG"

# Use a model by full ID
cli core llm -i "Explain RAG" -m gpt41mini@openai

# Use a raw provider model name (resolved via models.dev fuzzy match)
cli core llm -i "Explain RAG" -m gpt-4o-mini

# Use a named tag
cli core llm -i "Explain RAG" -m fast_model

# Stream output
cli core llm -i "Explain RAG" -m gpt41mini@openai --stream
```

In Python:
```python
from genai_tk.core.factories.llm_factory import get_llm

llm = get_llm()  # default from config
llm = get_llm("gpt41mini@openai")  # explicit model
llm = get_llm("fast_model")  # named tag
```

---

## Inspecting available models

### `cli info config`
Shows the active default model, all configured tags, and API key availability:

```bash
cli info config
```

### `cli info models`
Lists every provider, the number of known models per provider, and which API keys
are set:

```bash
cli info models
```

### `cli info llm-profile <model-id>`
Shows the full capability profile for a model — context window, max output tokens,
vision/reasoning/tool-call flags, per-token cost, and regions when available — sourced
from the local model catalogues with any YAML overrides applied:

```bash
cli info llm-profile gpt41mini@openai
cli info llm-profile gpt-4o-mini          # raw provider name also works
cli info llm-profile glm-5.2@edenai
cli info llm-profile glm-5.2-cloudflare@edenai
cli info llm-profile --reload             # refresh local models.dev database
```

---

## The `models.dev` database

`data/models_dev.json` is a local cache of the [models.dev](https://models.dev)
registry — an open database of 100+ AI providers and their models with structured
metadata (context window, pricing, capabilities, API endpoint, etc.).

It is used to:
- **Auto-resolve capabilities** (vision, reasoning, structured output, PDF support)
  when they are not set explicitly in `llm.yaml`
- **Provide cost estimates** (`Cost in / Cost out` in `llm-profile`)
- **Drive fuzzy name resolution** — e.g. `gpt-4o-mini` maps to the canonical
  `gpt-4o-mini-2024-07-18` entry

### Refreshing the database

```bash
cli info llm-profile --reload      # downloads latest models.dev and exits
```

The file is stored at `data/models_dev.json` relative to the project root.

### EdenAI catalogue

EdenAI models are fetched from authenticated `/v3/models` endpoints and cached
beside the models.dev catalogue. `edenai` uses `https://api.edenai.run/v3/models`
and `data/edenai_models.json`; `edenai-eur` uses
`https://api.eu.edenai.run/v3/models` and `data/edenai_eur_models.json`. Both
requests use `EDENAI_API_KEY`. The European catalogue contains only models eligible
for EU inference and is maintained separately from the global catalogue.

EdenAI metadata is normalized into the same model profile used for models.dev. This
includes input and output modalities, reasoning, function calling, structured output,
context length, and token pricing. The `llm-profile` command derives `vision`, `pdf`,
`audio`, and `video` from the model's advertised input modalities. A capability is shown
only when EdenAI reports it for the selected inference route.

EdenAI can expose one model through several inference backends. Fuzzy resolution
groups all backends for the selected model before similar model versions, so
`glm-5.2@edenai` shows each available GLM-5.2 route before GLM-5.1 alternatives.
An explicit backend suffix, such as `glm-5.2-cloudflare@edenai`, selects that
backend first. The CLI displays at most 20 fuzzy-resolution candidates.

Use `@edenai-eur` to resolve and call only EU-eligible routes. For example,
`glm-5.2-scaleway@edenai-eur` resolves against the European catalogue and sends
the request to EdenAI's European endpoint. The provider suffix controls this
selection:

```bash
# Global EdenAI catalogue and endpoint
cli core llm -i "Tell me a joke" -m glm-5.2-scaleway@edenai

# EU-filtered EdenAI catalogue and European inference endpoint
cli core llm -i "Tell me a joke" -m glm-5.2-scaleway@edenai-eur
```

### Overriding capability data

If the database entry is absent or incorrect for your use case, override it in
`llm.yaml`:

```yaml
- model_id: my_model
  providers:
    - openai: my-model-name
  capabilities: [vision, structured_outputs]
  max_tokens: 8192
  context_window: 128000
```

---

## Adding a new provider

1. Add the provider configuration to `config/providers/providers.yaml`
2. Add model entries to `config/providers/llm.yaml`
3. Set the required API key in your `.env`
4. Verify with `cli info models` and `cli info llm-profile <new-model-id>`

For providers not yet in `models.dev`, set `capabilities`, `max_tokens`, and
`context_window` explicitly in `llm.yaml` to avoid missing metadata warnings.
