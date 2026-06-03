# Configuration

The configuration system is based on [OmegaConf](https://omegaconf.readthedocs.io/) with hierarchical YAML files, environment variable substitution, and gitignore-style file discovery.

## How it works

1. On startup, `global_config()` walks up from the current working directory to find `config/app_conf.yaml`.
2. `app_conf.yaml` is loaded first and sets the **profile** name (default: `local`).
3. The optional **`:merge:`** key lists gitignore-style patterns — all matching `*.yaml` files relative to `config/` are deep-merged in sorted order.
4. The optional **`:profiles:`** key contains an inline dict whose keys are profile names. The block matching the active profile is deep-merged last.
5. Both `:merge:` and `:profiles:` pseudo-keys are stripped from the final config.
6. `${oc.env:VAR,default}` expressions are resolved from environment (or `.env` file in project root).

This works correctly whether you run from the project root, a notebook, a subdirectory, or a deployed container.

## File layout

```
config/
├── app_conf.yaml               # Entry point: :merge: patterns, :profiles: block, paths
├── agents/
│   ├── langchain/              # LangChain agent profiles (dict-based)
│   │   ├── defaults.yaml
│   │   ├── simple.yaml
│   │   ├── deep.yaml
│   │   ├── browser.yaml
│   │   └── text2sql.yaml
│   ├── deepagent/              # Deep agents (dict-based)
│   │   ├── global.yaml
│   │   ├── coder.yaml
│   │   ├── fast.yaml
│   │   └── researcher.yaml
│   └── deerflow.yaml           # Deer-flow agent profiles
├── providers/
│   ├── llm.yaml                # LLM model declarations
│   └── embeddings.yaml         # Embeddings model declarations
├── rag.yaml                    # RAG, chunkers, embeddings stores, retrievers
├── workflows.yaml              # Workflow definitions
├── mcp_servers.yaml            # MCP server configurations
└── README.md                   # This directory's structure
```

There is no `profiles/` directory or `overrides.yaml` — profile-specific values live in the `:profiles:` block inside `app_conf.yaml`, and local overrides can be added there or via an explicit entry in `:merge:`.

## `app_conf.yaml` — entry point

```yaml
# Profile selects which :profiles: block key is merged.
# Override with GENAITK_PROFILE env var.
profile: ${oc.env:GENAITK_PROFILE,local}

# Gitignore-style patterns relative to this config/ directory.
# All matching *.yaml files (except app_conf.yaml itself) are deep-merged in sorted order.
:merge:
  - "**/*.yaml"    # include everything
  - "!demos/**"    # exclude demos/
  - "!mcp/**"      # exclude mcp/

# Optional: set environment variables at startup (no-op if already set).
:env:
  LOGURU_LEVEL: INFO
  LANGCHAIN_TRACING_V2: "false"

paths:
  home: ${oc.env:HOME}
  project: ${oc.env:PWD}
  config: ${paths.project}/config
  data_root: ${paths.project}/data

test:
  unit: tests/unit_tests
  integration: tests/integration_tests

cli:
  commands:
    - genai_tk.cli.commands_core.CoreCommands
    # ... more commands

# Profile-specific overlays — the key matching `profile` is deep-merged last.
:profiles:
  local:
    llm:
      models:
        default: gpt_oss120@openrouter
    llm_cache:
      method: sqlite

  pytest:
    llm:
      models:
        default: parrot_local@fake
    llm_cache:
      method: memory

  prod:
    llm:
      models:
        default: gpt_4o@openai
    monitoring:
      langsmith: true
```

**Key pseudo-keys** (stripped before the final config is exposed):

| Key | Purpose |
|---|---|
| `:merge:` | List of gitignore-style patterns; matching files are deep-merged |
| `:profiles:` | Dict of profile overlays; the key matching `profile` is applied |
| `:env:` | Dict of env vars to set at startup (no-op if already in environment) |

## `:merge:` — file discovery

Patterns are **gitignore-style** (via [pathspec](https://pypi.org/project/pathspec/)), resolved relative to the directory containing `app_conf.yaml`.

```yaml
:merge:
  - "**/*.yaml"      # recursive — all yaml files
  - "!demos/**"      # exclude the demos/ subtree
  - "!mcp/**"        # exclude mcp/ subtree
  - "overrides.yaml" # explicit single file (if you want a local-only override file)
```

Rules:
- `!` prefix **excludes** matching paths (like `.gitignore`)
- Files are processed in **sorted** order
- `app_conf.yaml` itself is always skipped
- `:profiles:` in merged files is ignored (only `app_conf.yaml`'s `:profiles:` block is honoured)

## `:profiles:` — inline profile overlays

Instead of a `profiles/` directory, profile-specific config lives directly in `app_conf.yaml` under `:profiles:`. The key matching the active `profile` value is deep-merged last.

```yaml
profile: ${oc.env:GENAITK_PROFILE,local}

:profiles:
  local:
    llm:
      models:
        default: gpt_oss120@openrouter
    llm_cache:
      method: sqlite

  pytest:
    llm:
      models:
        default: parrot_local@fake
    llm_cache:
      method: memory
    vector_store:
      postgres_url: null

  test_unit:
    # Named contexts for unit test use via use_context()
    test_env:
      llm:
        models:
          default: gpt-3.5-turbo
    prod_env:
      llm:
        models:
          default: gpt-4
```

The `:profiles:` block can also include a nested `:merge:` list to pull in additional files for a specific profile:

```yaml
:profiles:
  prod:
    :merge:
      - "prod_secrets.yaml"   # loaded only when profile=prod
    llm:
      models:
        default: gpt_4o@openai
```

### Switching profiles

```bash
# Load 'local' profile (default)
python myapp.py

# Load 'prod' profile
GENAITK_PROFILE=prod python myapp.py

# Load 'pytest' profile (used by conftest.py automatically)
GENAITK_PROFILE=pytest python myapp.py
```

In Python:
```python
from genai_tk.utils.config_mngr import switch_profile
switch_profile("prod")    # reloads all config with profile=prod
```

### Two-layer system

| Layer | Mechanism | When to use |
|---|---|---|
| **Profile** | `GENAITK_PROFILE=pytest` → applies `:profiles: pytest:` block | Deployment environment (local, prod, ci, docker, pytest) |
| **Context** | `global_config().use_context("training_local")` | In-session overlay from a named top-level key, no file reload |

`switch_profile()` reloads all config files. `use_context()` merges a named top-level key as an overlay without reloading.

## OmegaConf merge behavior

- **Dicts are deep-merged**: nested keys from later files extend earlier files
- **Lists are replaced**: later file's list replaces earlier file's list entirely
- **Scalars are replaced**: later values win

### Avoiding conflicts

Use **dict-keyed structure** instead of lists to avoid conflicts:

**❌ Wrong** (causes warnings):
```yaml
langchain_agents:
  profiles:           # ← List replacement conflict!
    - name: research
```

**✅ Right** (no conflicts):
```yaml
langchain_agents:
  research:           # ← Dict key, merged together
    name: research
```

## Agent configuration

Agent profiles are stored as **dict-keyed entries**, not lists. The **key** is what you pass to `--profile` CLI flag; the `name` field is display name only.

### LangChain agents

Files: `config/agents/langchain/defaults.yaml`, `simple.yaml`, `deep.yaml`, `browser.yaml`, `text2sql.yaml`

```yaml
langchain_agents:
  default_profile: "simple"    # Used when --profile not specified
  
  defaults:                    # Inherited by all profiles
    type: react
    llm: null
    
  simple:                      # ← Profile KEY (used as: -p simple)
    name: "simple"             # Display name only
    type: react
    tools: [...]
    
  research:                    # ← Another profile key
    name: "Research"           # Display name (different from key)
    type: deep
    skill_directories: [...]
```

**CLI usage:**
```bash
cli agents langchain --list              # List all profiles (shows both key and name)
cli agents langchain -p simple "Hello"   # Use 'simple' profile (by KEY)
cli agents langchain -p research --chat  # Use 'research' profile (by KEY)
```

### Deep agents

Files: `config/agents/deepagent/global.yaml`, `coder.yaml`, `researcher.yaml`, `fast.yaml`

```yaml
deepagent:
  default_profile: null        # No default profile
  
  coder:                       # ← Profile KEY
    name: coder               # Display name
    llm: gpt_4@openai
    enable_planning: true
    
  researcher:                  # ← Another key
    name: researcher
    llm: claude-opus@openrouter
```

**CLI usage:**
```bash
cli agents deepagent --list              # List all profiles
cli agents deepagent --profile coder     # Use 'coder' profile (by KEY)
cli agents deepagent task --profile researcher "Analyze this..."
```

## Workflow configuration

File: `config/workflows.yaml`

Workflows are defined as YAML with step templates and execution profiles that bind values.

See [workflows.md](workflows.md) for complete workflow documentation.

## RAG configuration

File: `config/rag.yaml`

Contains embeddings stores, chunkers, retrievers, and RAG pipeline configuration.

See [rag.md](rag.md) for complete RAG documentation.

## Accessing config in Python

```python
from genai_tk.utils.config_mngr import global_config, switch_profile, use_active_context

config = global_config()                        # singleton, auto-discovered
value = config.get("llm.models.default")        # dot-separated key path

# LLM runtime selection examples
from genai_tk.core.factories.llm_factory import get_llm

# Inline effort on model identifier
llm = get_llm(llm="gpt-oss-120b (high)@openrouter")

# Explicit reasoning payload (preferred for provider-specific options)
llm = get_llm(
  llm="gpt-oss-120b@openrouter",
  reasoning={"effort": "high", "resume": "cursor-token", "max_tokens": 4096},
)

# Switch the active deployment profile (reloads all config files)
switch_profile("prod")                          # set GENAITK_PROFILE=prod + reload
switch_profile("pytest")                        # use fake models for tests

# Activate a named context overlay (no file reload, lightweight)
config.use_context("training_local")            # merge training_local: sub-dict on top
config.use_context("training_openai")           # switch to openai variant
```

## Debugging configuration

### View loaded config

```bash
cli info config-keys                 # Show top-level keys and their source files
cli info config-keys llm             # Drill down into llm config
cli info config-keys llm.models      # Go deeper
```

### View merged config

```bash
python -c "from genai_tk.utils.config_mngr import global_config; import yaml; print(yaml.dump(dict(global_config())))" > /tmp/merged_config.yaml
```

## Environment variables (`.env`)

Place a `.env` file in the project root (or any parent directory). It is loaded automatically at startup before any config values are resolved.

```bash
# .env — do not commit
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
GENAITK_PROFILE=local  # optional: select profile
```

## Testing profiles

Two built-in profiles serve the test suite (defined inline in `app_conf.yaml`'s `:profiles:` block):

| Profile | Purpose |
|---|---|
| `pytest` | Fake models, memory caches — activated by `conftest.py` via `switch_profile("pytest")` |
| `test_unit` | `test_env:` and `prod_env:` context targets used by config-manager unit tests |

The test suite switches profiles automatically — no manual env var needed.

## `cli init` — initializing a new project

`cli init` copies the bundled default config into the current directory. It works before any config exists.

```bash
uv run cli init                       # copy config/ to ./config/
uv run cli init --force               # overwrite existing files
```

After running, edit `.env` with your API keys.

## See also

- [agents.md](agents.md) — Agent profiles and configuration
- [rag.md](rag.md) — RAG configuration and retrieval
- [workflows.md](workflows.md) — Workflow engine and orchestration
- [cli.md](cli.md) — CLI commands and `cli info config-keys`

