# Copilot Instructions for My AI App

This is a **genai-tk** based AI application project.

## Key Rules

- Use `uv` to run Python and manage dependencies (not pip directly)
- Use **Pydantic v2** for all data models — never dataclasses
- Use **absolute imports** only: `from my_ai_app.commands import ...`
- Use modern Python 3.12+ syntax: `str | None`, `list[str]`
- Format with **ruff** (line-length 120)
- Configuration uses **OmegaConf** YAML with `${oc.env:VAR}` interpolation

## Core Imports

```python
from genai_tk.core.llm_factory import get_llm          # Create LLM instances
from genai_tk.core.prompts import def_prompt            # Build prompts
from genai_tk.cli.base import CliTopCommand             # CLI command groups
from genai_tk.utils.config_mngr import global_config    # Access configuration
from genai_tk.core.chain_registry import register_runnable  # Register chains
```

## Where to Put Things

| What | Where | Register in |
|------|-------|-------------|
| CLI commands | `my_ai_app/commands/` | `config/app_conf.yaml` → `cli.commands` |
| LCEL chains | `my_ai_app/chains/` | Call `register_runnable()` in the module |
| Webapp pages | `my_ai_app/webapp/pages/` | `config/webapp.yaml` → `ui.navigation` |
| Agent profiles | — | `config/agents/langchain.yaml` |
| Tools | `my_ai_app/tools/` | Reference from agent profile YAML |

## Reference

See `AGENTS.md` for detailed coding guidelines, patterns, and genai-tk documentation links.
