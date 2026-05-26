---
name: genai-tk-cli-and-scaffolding
description: Add or modify genai-tk Typer CLI commands, dynamic command registration, project scaffolding, and generated Copilot/agent support files.
---

# GenAI Toolkit CLI And Scaffolding

## Read First

- `docs/cli.md`
- `docs/scaffolding.md`
- `docs/copilot-agent-support.md`
- `genai_tk/main/cli.py`
- `genai_tk/main/scaffolder.py`
- `genai_tk/cli/base.py`
- `config/app_conf.yaml`

## Code Map

| Concern | Paths |
|---|---|
| Entry point | `genai_tk/main/cli.py`, `[project.scripts]` in `pyproject.toml` |
| Command group base | `genai_tk/cli/base.py` |
| Built-in command groups | `genai_tk/cli/`, `genai_tk/agents/**/commands*.py`, `genai_tk/workflow/commands.py`, `genai_tk/mcp/cli_commands.py` |
| Dynamic registration | `config/app_conf.yaml` under `cli.commands` |
| Scaffold generation | `genai_tk/main/commands_init.py`, `genai_tk/main/scaffolder.py` |
| Command tree tests | `tests/unit_tests/cli/test_command_tree.py` |

## Add A Command Group

1. Create a class deriving from `genai_tk.cli.base.CliTopCommand`.
2. Implement `get_description()` and `register_sub_commands()`.
3. Register the fully qualified class path in `config/app_conf.yaml`.
4. Add CLI tests under `tests/unit_tests/cli/` or the owning domain test folder.

## Commands

```bash
uv run cli --help
uv run cli <group> --help
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/cli -q
```

## Avoid

- Do not import heavy optional dependencies at module import time; lazy-load inside command functions.
- Do not hardcode command groups in `main.cli` when config registration is intended.
- Do not make scaffold output diverge from docs without updating `docs/scaffolding.md`.
