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
- **Auto-scan**: All `*.yaml` files in `config/` are merged, then profile overlay from `config/profiles/<profile>/`, then `config/overrides.yaml`
- **Profile overlays**: `config/profiles/local/`, `config/profiles/prod/`, etc.
- **Dict-keyed profiles**: Agent profiles are dicts with keys (e.g., `research`, `coding`), not lists

### Quick Config Access

```python
config = global_config()
config.get("llm.models.default")                  # Access nested keys
config.get("langchain_agents.research.type")      # Agent profile access
```

### Selecting Profiles at Runtime

```bash
# In CLI
cli agents langchain -p research "Your query"      # Use 'research' profile KEY
cli agents langchain --list                        # Show all profiles

# In environment
GENAITK_PROFILE=prod python myapp.py              # Load config/profiles/prod/

# In Python
from genai_tk.agents.langchain.config import resolve_profile
profile = resolve_profile(config, "research")     # Profile KEY (case-insensitive)
```

## Where to Put Things

| What | Where | Configuration |
|------|-------|---|
| CLI commands | `genai_tk/cli/` | `config/app_conf.yaml` → `cli.commands` |
| LCEL chains | `genai_tk/chains/` | Call `register_runnable()` in module |
| Agent profiles (LangChain) | — | `config/agents/langchain/*.yaml` (dict-keyed) |
| Agent profiles (Deep) | — | `config/agents/deepagent/*.yaml` (dict-keyed) |
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
