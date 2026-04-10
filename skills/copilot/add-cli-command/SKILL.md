---
name: add-cli-command
description: Step-by-step procedure to create a new CLI command group in a genai-tk project and register it in configuration.
---

# Add a CLI Command Group

Follow these steps to add a new CLI command group to a genai-tk project.

## Prerequisites
- The project was initialized with `cli init` (has `config/app_conf.yaml`)
- The project has a Python package directory with a `commands/` sub-directory

## Step 1: Create the Command Class

Create a new file in `<package>/commands/my_commands.py`:

```python
"""My custom CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from genai_tk.cli.base import CliTopCommand

console = Console()


class MyCommands(CliTopCommand):
    """Description of what these commands do."""

    description: str = "Short description for --help output"

    def get_description(self) -> tuple[str, str]:
        # First element is the command group name (used as: cli <name> <subcommand>)
        return "mygroup", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def subcommand_one(
            arg: Annotated[str, typer.Argument(help="Positional argument")],
            flag: Annotated[bool, typer.Option("--flag", "-f", help="Optional flag")] = False,
        ) -> None:
            """One-line description of the subcommand."""
            console.print(f"Running with {arg}, flag={flag}")
```

## Step 2: Register in Configuration

Add the fully-qualified class path to `config/app_conf.yaml` under `cli.commands`:

```yaml
cli:
  commands:
    # ... existing commands ...
    - <package_name>.commands.my_commands.MyCommands
```

## Step 3: Verify

```bash
uv run cli mygroup --help
uv run cli mygroup subcommand-one "test"
```

## Key Patterns

- The command group name in `get_description()` becomes the CLI sub-command: `cli <name>`
- Use `Annotated[type, typer.Argument(...)]` for positional args
- Use `Annotated[type, typer.Option(...)]` for flags/options
- Import heavy dependencies inside the function body (lazy loading)
- Use `rich.console.Console` for formatted output
- If you need config access: `from genai_tk.utils.config_mngr import global_config`
- If you need LLM access: `from genai_tk.core.llm_factory import get_llm`
