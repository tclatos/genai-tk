# Configuration

The configuration system is based on [OmegaConf](https://omegaconf.readthedocs.io/) with hierarchical YAML files, environment variable substitution, and auto-discovery from parent directories.

## How it works

1. On startup, `global_config()` walks up from the current working directory to find `config/app_conf.yaml`.
2. `app_conf.yaml` is loaded first and sets the **profile** name (default: `baseline`).
3. **Auto-scan** loads all `*.yaml` files from `config/` (recursively), sorted alphabetically:
   - **Base files** — auto-loaded from `config/` root and subdirectories
   - **Profile overlay** — auto-loaded from `config/profiles/<profile_name>/` (only if selected)
   - **Overrides** — `config/overrides.yaml` loaded last (wins over everything)
4. All files are **deep-merged** in order — dict values are merged, lists are replaced (OmegaConf behavior).
5. `${oc.env:VAR,default}` expressions are resolved from environment (or `.env` file in project root).

This works correctly whether you run from the project root, a notebook, a subdirectory, or a deployed container.

## File layout

```
config/
├── app_conf.yaml               # Entry point: profile selection, paths
├── profiles/
│   └── baseline/               # Profile overlay — loaded when profile=baseline
│       └── baseline.yaml       # Default LLM, embeddings, cache, vector store
├── overrides.yaml              # Local overrides (loaded last, wins)
├── agents/
│   ├── langchain/              # LangChain agent profiles (dict-based)
│   │   ├── defaults.yaml       # Default settings and global config
│   │   ├── simple.yaml         # React agent profiles (simple, filesystem, weather)
│   │   ├── deep.yaml           # Deep agents (research, coding, data_analysis, etc.)
│   │   ├── browser.yaml        # Browser automation agents
│   │   └── text2sql.yaml       # Text-to-SQL agents
│   ├── deepagent/              # Deep agents (dict-based)
│   │   ├── global.yaml         # Global settings
│   │   ├── coder.yaml          # Coder profile
│   │   ├── fast.yaml           # Fast profile
│   │   └── researcher.yaml     # Researcher profile
│   └── deerflow.yaml           # Deer-flow agent profiles
├── providers/
│   ├── llm.yaml                # LLM model declarations
│   └── embeddings.yaml         # Embeddings model declarations
├── rag.yaml                    # RAG, chunkers, embeddings stores, retrievers
├── workflows.yaml              # Workflow definitions
├── mcp_servers.yaml            # MCP server configurations
└── README.md                   # This directory's structure
```

## `app_conf.yaml` — entry point

```yaml
# Profile selects a profile overlay directory: config/profiles/<profile>/
# Override with BLUEPRINT_CONFIG env var.
profile: ${oc.env:BLUEPRINT_CONFIG,baseline}

# Directories scanned recursively for YAML files at startup.
# Defaults to [config/] when omitted. All *.yaml files are merged (sorted),
# except app_conf.yaml, profiles/, and anything in config_exclude.
# config_dirs: [config/]

# Glob patterns (relative to each config_dir) to skip during auto-scan.
# config_exclude: []

:env:
  LOGURU_LEVEL: INFO
  LANGCHAIN_TRACING_V2: "false"
  DEER_FLOW_PATH: ${oc.env:DEER_FLOW_PATH,${paths.project}/ext/deer-flow}

paths:
  home: ${oc.env:HOME}
  project: ${oc.env:PWD}
  config: ${paths.project}/config
  data_root: ${paths.project}/data

test:
  unit: tests/unit_tests
  integration: tests/integration_tests
  evals: tests/eval_tests
  notebooks: examples/notebooks

cli:
  commands:
    - genai_tk.cli.commands_core.CoreCommands
    - genai_tk.cli.commands_info.InfoCommands
    # ... more commands
```

**Key fields:**
- `profile` — selects which `config/profiles/<profile>/` overlay is loaded. Default: `baseline`
- `:env` — environment variables to set at startup (does not override existing env vars)
- `${oc.env:VAR,default}` — reads `VAR` from environment, falls back to `default`
- `paths.*` — auto-detected at runtime; absolute paths used by all components

## Profile overlays

A **profile** is a directory under `config/profiles/<name>/` that is loaded on top of base configuration. This allows environment-specific configs (dev, staging, prod) without duplicating common settings.

### Using profiles

**At startup** (automatic):
```bash
# Load config/profiles/baseline/ (default)
python myapp.py

# Load config/profiles/prod/
BLUEPRINT_CONFIG=prod python myapp.py

# Load config/profiles/custom/
BLUEPRINT_CONFIG=custom python myapp.py
```

### Creating a profile

```bash
mkdir -p config/profiles/prod
```

Add any YAML files to override base configuration:
```yaml
# config/profiles/prod/llm.yaml
llm:
  models:
    default: gpt_4o@openai  # Use GPT-4 in production

# config/profiles/prod/embeddings.yaml
embeddings:
  models:
    default: ada_002@openai  # Use Ada in production
```

Only the values you override are merged; base configuration remains in place.

## Auto-scan loading order

Files are loaded in this order:

1. **Base files** from `config/` (and subdirectories), sorted alphabetically
2. **Profile overlay** from `config/profiles/<profile>/` (alphabetically)
3. **Overrides** — `config/overrides.yaml` (loaded last, wins)

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
from genai_tk.utils.config_mngr import global_config

config = global_config()                        # singleton, auto-discovered
value = config.get("llm.models.default")        # dot-separated key path
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
BLUEPRINT_CONFIG=baseline  # optional: select profile
```

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
