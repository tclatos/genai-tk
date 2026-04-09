# Configuration

The configuration system is based on [OmegaConf](https://omegaconf.readthedocs.io/) with hierarchical YAML files, environment variable substitution, and auto-discovery from parent directories.

## How it works

1. `cli init` (or `git clone`) places a `config/` directory next to your project root.
2. On startup, `global_config()` walks up from the current working directory until it finds `config/app_conf.yaml`.
3. `app_conf.yaml` declares a `:merge` list — all listed YAML files are loaded and deep-merged in order.
4. Any `${oc.env:VAR,default}` expressions are resolved from the environment (and from `.env` in the project root).

This means the config works correctly whether you run from the project root, a notebook, a subdirectory, or a deployed container.

## File layout

```
config/
├── app_conf.yaml               # Entry point: sets default_config and :merge list
└── basic/                      # Default environment (pointed to by app_conf.yaml)
    ├── init/
    │   ├── baseline.yaml       # Default LLM, embeddings, cache, vector store
    │   ├── overrides.yaml      # Local or per-deployment overrides (git-ignored)
    │   └── mcp_servers.yaml    # MCP server definitions
    ├── providers/
    │   ├── llm.yaml            # LLM model declarations
    │   ├── embeddings.yaml     # Embeddings model declarations
    │   └── providers.yaml      # Provider API key env vars and class mappings
    └── agents/
        ├── langchain.yaml      # LangChain agent profiles
        └── deerflow.yaml       # Deer-flow profiles
```

Agent and demo configs are **not** merged globally — they are loaded on demand by their respective loaders.

## `app_conf.yaml` — entry point

```yaml
default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}

:merge:
  - ${paths.config}/init/baseline.yaml
  - ${paths.config}/providers/llm.yaml
  - ${paths.config}/providers/embeddings.yaml
  - ${paths.config}/init/overrides.yaml
  - ${paths.config}/init/mcp_servers.yaml

:env:
  LOGURU_LEVEL: INFO
  LANGCHAIN_TRACING_V2: "false"
  DEER_FLOW_PATH: ${oc.env:DEER_FLOW_PATH,${paths.project}/ext/deer-flow}

paths:
  home: ${oc.env:HOME}
  project: ${oc.env:PWD}   # auto-detected at runtime
  config: ${paths.project}/config
  data_root: ${paths.project}/data
```

**Directives:**
- `:merge` — list of YAML files to deep-merge into the root config. Later files win.
- `:env` — environment variables to set at startup (do not override existing env vars).
- `${oc.env:VAR,default}` — reads `VAR` from the environment, falls back to `default`.

## `baseline.yaml` — defaults

Sets the default LLM, embeddings model, cache backend, and vector store used when no override is specified:

```yaml
llm:
  models:
    default: gpt_oss120@openrouter
    cheap_model: claude-haiku@openrouter
    fast_model: claude-haiku@openrouter
    fake: parrot_local@fake      # no API key — useful for tests
  cache: sqlite
  cache_path: data/llm_cache/langchain.db

embeddings:
  models:
    default: ada_002@openai
    local: artic_22@ollama
    fake: embeddings_768@fake
```

## `providers/llm.yaml` — LLM model declarations

Each entry maps a logical `model_id` to one or more provider+model-name pairs.
The runtime selects the first provider whose API key is available.

```yaml
llm:
  exceptions:
    - model_id: gpt41mini
      providers:
        - openai: gpt-4.1-mini-2025-04-14    # direct provider
    - model_id: haiku
      providers:
        - openrouter: anthropic/claude-haiku-4-5  # via gateway
    - model_id: parrot_local
      providers:
        - fake: parrot                        # built-in echo model
```

Use `model_id@provider` format at runtime — see [docs/llm-selection.md](llm-selection.md).

## `providers/embeddings.yaml` — embeddings declarations

Same structure as `llm.yaml` but for embedding models:

```yaml
embeddings:
  exceptions:
    - model_id: ada_002
      providers:
        - openai: text-embedding-ada-002
    - model_id: artic_22
      providers:
        - ollama: snowflake-arctic-embed2:22m
    - model_id: embeddings_768
      providers:
        - fake: dim768                        # for testing
```

## `agents/langchain.yaml` — agent profiles

Agent profiles are loaded on demand by `LangchainAgent.from_profile()` and `cli agents langchain -p <name>`.

```yaml
langchain_agents:
  defaults:
    type: react
    checkpointer: {type: none}
  default_profile: Simple
  profiles:
    - name: Simple
      type: react
      llm: fast_model
      system_prompt: "You are a helpful assistant."

    - name: Coding
      type: deep
      llm: gpt41mini@openai
      system_prompt: "You are an expert Python developer."
      tools: [python_repl, file_system]
```

See [docs/agents.md](agents.md) for all profile fields and how to add tools.

## Environments

You can have multiple named environments in one `app_conf.yaml` by adding extra keys alongside `baseline`:

```yaml
default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}
baseline:
  # ... (merged from :merge list)
production:
  llm:
    cache: redis
    models:
      default: gpt41@openai
```

Switch environment at runtime:
- **Environment variable**: `export BLUEPRINT_CONFIG=production`
- **Python**: `global_config().select_config("production")`

## Environment variables (`.env`)

Place a `.env` file in the project root (or any parent directory). It is loaded automatically at startup before any config values are resolved.

```bash
# .env — do not commit
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
BLUEPRINT_CONFIG=production   # optional: switch environment
```

## Accessing config in Python

```python
from genai_tk.utils.config_mngr import global_config

config = global_config()                        # singleton, auto-discovered
value = config.get("llm.models.default")        # dot-separated key path
config.set("llm.models.default", "haiku")       # runtime override
config.select_config("production")              # switch named environment
```

## `cli init` — initializing a new project

`cli init` copies the bundled default config into the current directory. It works before any config exists.

```bash
uv run cli init                       # copy config/ to ./config/
uv run cli init --force               # overwrite existing files
uv run cli init --deer-flow           # also clone + install Deer-flow backend
uv run cli init --deer-flow --force   # update existing Deer-flow clone
```

After running, edit `.env` with your API keys and `config/providers/llm.yaml` to declare your models.

## See also

- [LLM model IDs, tags, and the models.dev database](llm-selection.md)
- [Agent profile fields](agents.md#configuration-system)
- [Deer-flow profiles](deer-flow.md#profiles)
