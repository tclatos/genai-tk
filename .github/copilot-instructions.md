# Copilot Instructions for GenAI Toolkit

This is a **genai-tk** based AI application project.

## Key Rules

- Use `uv` to run Python and manage dependencies (not pip directly)
- Use **Pydantic v2** for all data models — never dataclasses
- Use **absolute imports** only: `from genai_tk.agents import ...`
- Use modern Python 3.12+ syntax: `str | None`, `list[str]`
- Format with **ruff** (line-length 120)
- Configuration uses **OmegaConf** YAML with `${oc.env:VAR}` interpolation and profile overlays — see skill `configuration` for access patterns
- Agent profiles are dict-keyed by profile **key** (e.g. `research`), not by display name — use the key in CLI commands (`cli agents run research`)

## Core Imports

```python
from genai_tk.core.factories import get_llm  # Create LLM instances
from genai_tk.core.prompts import def_prompt  # Build prompts
from genai_tk.cli.base import CliTopCommand  # CLI command groups
from genai_tk.config_mgmt.config_mngr import global_config  # Access configuration
from genai_tk.agents.langchain.config import resolve_profile  # Load agent profile by key
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

## Reference

See `AGENTS.md` for detailed coding guidelines, the config access patterns, agent profile YAML format, and the full skill index (`skills/README.md`).
