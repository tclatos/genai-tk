---
name: genai-tk-configuration
description: Work on genai-tk OmegaConf configuration, profiles, overrides, env substitution, and config discovery. Use when editing config/*.yaml or genai_tk.utils.config_mngr.
---

# GenAI Toolkit Configuration

## Read First

- `docs/configuration.md`
- `genai_tk/utils/config_mngr.py`
- `genai_tk/utils/config_exceptions.py`
- `tests/unit_tests/utils/test_config_mngr.py`

## Core Model

Configuration starts from `config/app_conf.yaml`, then auto-scans YAML files under `config/`, applies `config/profiles/<profile>/`, and finally applies `config/overrides.yaml` when present.

Use these paths as the source of truth:

| Concern | Path |
|---|---|
| Entry point, profile, paths, CLI command classes | `config/app_conf.yaml` |
| Local/default model settings | `config/profiles/local/genai_def.yaml` |
| Test fake model settings | `config/profiles/pytest/genai_def.yaml` |
| Providers | `config/providers/llm.yaml`, `config/providers/embeddings.yaml` |
| Agents | `config/agents/` |
| RAG | `config/rag.yaml` |
| Workflows | `config/workflows.yaml` |
| Webapp | `config/webapp.yaml` |

## Change Workflow

1. Add config under the domain-specific YAML file rather than hardcoding defaults in Python.
2. Use `${oc.env:VAR,default}` for external values and secrets.
3. Keep profile overlays small and deployment-specific.
4. If introducing a new config schema, model it with Pydantic v2 and validate at load boundaries.
5. Add a unit test that switches to the `pytest` profile or loads a temporary config tree.

## Commands

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/utils/test_config_mngr.py -q
uv run cli info config
```

## Avoid

- Do not read environment variables deep inside domain code when the value belongs in config.
- Do not append to list values in profile overlays expecting OmegaConf to merge lists; lists replace.
- Do not add local secrets to versioned config. Use environment variables or ignored overrides.
