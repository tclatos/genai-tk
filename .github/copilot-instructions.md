# Copilot Instructions for GenAI Toolkit

This is a **genai-tk** based AI application project.

## Key Rules

- Use `uv` to run Python and manage dependencies (not pip directly)
- Use **Pydantic v2** for all data models — never dataclasses
- Use **absolute imports** only: `from genai_tk.agents import ...`
- Use modern Python 3.12+ syntax: `str | None`, `list[str]`
- Format with **ruff** (line-length 120)
- Configuration uses **OmegaConf** YAML with `${oc.env:VAR}` interpolation and profile overlays

## Core Imports

```python
from genai_tk.core.factories import get_llm              # Create LLM instances
from genai_tk.core.prompts import def_prompt            # Build prompts
from genai_tk.cli.base import CliTopCommand             # CLI command groups
from genai_tk.utils.config_mngr import global_config    # Access configuration
from genai_tk.agents.langchain.config import resolve_profile  # Load agent profile by key
```

## Configuration System

- **Entry point**: `config/app_conf.yaml` sets `profile` (default: `local`)
- **Auto-scan**: All `*.yaml` files in `config/` are merged in sorted order, then the matching `:profiles:` block is deep-merged last
- **Profile overlays**: defined inline in `app_conf.yaml` under `:profiles:` (not a separate directory)
- **Dict-keyed profiles**: Agent profiles are dicts with keys (e.g., `research`, `coding`), not lists

### Typed Config Access — Two Canonical Patterns

Every top-level YAML key should be read via a Pydantic model, not raw `.get()`:

**Case 1 — `section()` (single object):**
```python
from genai_tk.utils.config_mngr import global_config
from genai_tk.utils.prefect_server import PrefectConfig

cfg = global_config().section("prefect", PrefectConfig)
print(cfg.host, cfg.port)     # typed attributes
```

**Case 2 — `section_dict()` (dict of objects):**
```python
from genai_tk.core.embeddings_store import EmbeddingsStoreConfig

stores = global_config().section_dict("embeddings_store", EmbeddingsStoreConfig)
default = stores["default"]   # EmbeddingsStoreConfig, validated at load time
```

Both return empty model / empty dict when the key is absent — they never raise.

### Key Models Reference

| YAML key | Model | Location |
|---|---|---|
| `prefect` | `PrefectConfig` | `genai_tk.utils.prefect_server` |
| `cli` | `CliConfig` | `genai_tk.main.cli` |
| `sandbox` | `SandboxConfig` | `genai_tk.agents.sandbox.models` |
| `monitoring` | `MonitoringConfig` | `genai_tk.utils.tracing` |
| `auth` | `AuthConfig` | `genai_tk.utils.basic_auth` |
| `kv_store` (dict) | `KvStoreConfig` (union) | `genai_tk.extra.kv_store_registry` |
| `embeddings_store` (dict) | `EmbeddingsStoreConfig` | `genai_tk.core.embeddings_store` |
| `structured` (dict) | `StructuredConfig` | `genai_tk.extra.structured.baml_util` |
| `llm` | `LlmSection` | `genai_tk.core.factories.llm_factory` |
| `embeddings` | `EmbeddingsSection` | `genai_tk.core.factories.embeddings_factory` |
| `paths` | `PathsConfig` | `genai_tk.utils.config_mngr` |

### Adding a New Config Section

```python
# 1. Define the model
class MyConfig(BaseModel):
    host: str = "localhost"
    port: int = 8080

# 2. Add a typed accessor
def my_config() -> MyConfig:
    return global_config().section("my_section", MyConfig)
```

### Selecting Agent Profiles at Runtime

```bash
cli agents langchain -p research "Your query"      # Use 'research' profile KEY
cli agents langchain --list                        # Show all profiles
GENAITK_PROFILE=prod python myapp.py              # Load prod profile overlay
```

```python
from genai_tk.agents.langchain.config import resolve_profile
profile = resolve_profile(config, "research")     # Profile KEY (case-insensitive)
```

## Where to Put Things

| What | Where | Configuration |
|------|-------|---|
| CLI commands | `genai_tk/cli/` | `config/app_conf.yaml` → `cli.commands` (list of `QualifiedCallable`) |
| LCEL chains | `genai_tk/chains/` | Call `register_runnable()` in module |
| Agent profiles (LangChain) | — | `config/agents/langchain/*.yaml` (dict-keyed) |
| Tools | `genai_tk/agents/tools/` | Reference from agent YAML |
| Skills | `skills/custom/<name>/` | SKILL.md file + referenced by agent |
| MCP servers | — | `config/mcp_servers.yaml` |

## Agent Profile Structure (Dict-Keyed)

```yaml
# config/agents/langchain/deep.yaml
langchain_agents:
  research:               # ← Profile KEY (used as: -p research)
    name: "Research"      # ← Display name (shown in --list)
    type: deep
    llm: gpt_41@openai
    tools: [web_search]
```

**Important**: Use the **KEY** (lowercase) in CLI commands, not the **name**:
- ✅ `cli agents langchain -p research`
- ❌ `cli agents langchain -p Research`

## Reference

See `AGENTS.md` for detailed coding guidelines, patterns, and genai-tk documentation links.
