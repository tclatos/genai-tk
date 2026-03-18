"""GenAI Lab Command Line Interface.

This module provides the main entry point for the GenAI Lab CLI, offering commands to:
- Run and test LangChain Runnables with various configurations
- Get information about available chains and their schemas
- List available models (LLMs, embeddings, vector stores)
- Execute Fabric patterns for text processing
- Manage LLM configurations and caching

The CLI is built using Typer and supports:
- Interactive command completion
- Help documentation for all commands
- Configuration via environment variables and .env files
- Debug and verbose output modes
- Streaming and non-streaming execution

Command Registration
-------------------
The CLI supports two command registration mechanisms:

1. **Class-based (Recommended)**: Use `CliTopCommand` base class for structured command groups
2. **Function-based (Legacy)**: Use plain functions that accept a `typer.Typer` instance

Example - Class-based Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a command group class inheriting from `CliTopCommand`:

```python
from typing import Annotated
import typer
from genai_tk.cli.base import CliTopCommand

class MyCommands(CliTopCommand):
    '''My custom command group.'''

    description: str = "My custom command group description"

    def get_description(self) -> tuple[str, str]:
        '''Return command group name and description.'''
        return "mygroup", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        '''Register individual commands in this group.'''

        @cli_app.command()
        def hello(name: Annotated[str, typer.Argument(help="Name to greet")]) -> None:
            '''Say hello to someone.'''
            print(f"Hello, {name}!")

        @cli_app.command()
        def goodbye() -> None:
            '''Say goodbye.'''
            print("Goodbye!")
```

Register in `config/app_conf.yaml`:

```yaml
cli:
  commands:
    - mymodule.commands:MyCommands
```

Usage:

```bash
uv run cli mygroup hello World
# Output: Hello, World!

uv run cli mygroup goodbye
# Output: Goodbye!
```

Example - Function-based Registration (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a registration function:

```python
import typer


def register_commands(cli_app: typer.Typer) -> None:
    '''Register commands using function-based approach.'''

    my_app = typer.Typer(no_args_is_help=True, help="My command group")

    @my_app.command()
    def hello(name: str) -> None:
        '''Say hello.'''
        print(f"Hello, {name}!")

    cli_app.add_typer(my_app, name="mygroup")
```

Register in `config/app_conf.yaml`:

```yaml
cli:
  commands:
    - mymodule.commands:register_commands
```
"""

import sys

import typer
from dotenv import load_dotenv
from loguru import logger

from genai_tk.cli.base import CliTopCommand

# Import modules where runnables are registered
from genai_tk.utils.config_exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.utils.config_mngr import global_config, import_from_qualified
from genai_tk.utils.logger_factory import setup_logging

load_dotenv(verbose=True)


PRETTY_EXCEPTION = (
    False  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/
)

cli_app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=PRETTY_EXCEPTION,
)


def load_and_register_commands(cli_app: typer.Typer) -> None:
    """Load and register all CLI commands from config.

    Supports both class-based (CliTopCommand) and function-based registration.

    Args:
        cli_app: The Typer app instance to register commands to
    """
    try:
        modules = global_config().get_list("cli.commands", value_type=str)
    except ConfigKeyNotFoundError as e:
        logger.error(f"CLI commands configuration not found: {e.message}\nSuggestion: {e.suggestion}")
        raise typer.Exit(1) from e
    except ConfigTypeError as e:
        logger.error(
            f"Invalid CLI commands configuration type: {e.message}\n"
            f"Expected: {e.expected_type.__name__}, Got: {e.actual_type.__name__}\n"
            f"Suggestion: {e.suggestion}"
        )
        raise typer.Exit(1) from e
    except ConfigError as e:
        logger.error(f"Configuration error: {e.message}\nSuggestion: {e.suggestion}")
        raise typer.Exit(1) from e

    # Import and register commands from each module
    for module in modules:
        try:
            imported = import_from_qualified(module)

            # Check if it's a class or function
            if isinstance(imported, type):
                # It's a class - check if it's a CliTopCommand
                if issubclass(imported, CliTopCommand):
                    # Create instance and call register method
                    instance = imported()
                    instance.register(cli_app)
                else:
                    logger.warning(f"Class {imported.__name__} from {module} is not a CliTopCommand")
            else:
                # It's a function - call it directly (old API)
                imported(cli_app)
        except Exception as ex:
            logger.warning(f"Cannot load module {module}: {ex}")
            # Continue loading other modules instead of crashing


def register_commands(cli_app: typer.Typer) -> None:
    """Define additional utility commands for the CLI.

    Args:
        cli_app: The Typer app instance to add commands to

    Adds commands:
    - echo: Simple command to print a message
    - fabric: Execute Fabric patterns on input text
    """

    @cli_app.command()
    def echo(message: str) -> None:
        """Echo the message (for test purpose)"""
        print(message)


def main() -> None:
    """Main entry point for the GenAI Lab CLI.

    Handles logging setup, command registration, and error handling.
    """
    # We could do better with Typer @cli_app.callback(), but I haven't succeeded
    if "--logging" in sys.argv:
        level = "TRACE"
        sys.argv.remove("--logging")
    else:
        level = None

    try:
        setup_logging(level)
        load_and_register_commands(cli_app)
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        # Fatal configuration errors - display and exit
        logger.error(f"\n{'=' * 60}")
        logger.error(f"❌ Configuration Error: {e.__class__.__name__}")
        logger.error(f"{'=' * 60}")
        logger.error(f"Message: {e.message}")
        if hasattr(e, "suggestion") and e.suggestion:
            logger.error(f"\n💡 Suggestion: {e.suggestion}")
        logger.error(f"{'=' * 60}\n")
        raise typer.Exit(1) from e
    except ConfigError as e:
        # Generic configuration error
        logger.error(f"\n❌ Configuration error: {e.message}")
        if hasattr(e, "suggestion") and e.suggestion:
            logger.error(f"💡 Suggestion: {e.suggestion}")
        raise typer.Exit(1) from e
    except Exception as e:
        # Unexpected error during initialization
        logger.exception(f"Unexpected error during CLI initialization: {e}")
        raise typer.Exit(1) from e

    # Check if --help is requested or no arguments provided (show custom tree instead of default help)
    if len(sys.argv) == 1 or ("--help" in sys.argv and len(sys.argv) == 2):
        from genai_tk.cli.command_tree import display_command_tree

        display_command_tree(cli_app)
        return

    try:
        cli_app()
    except ConfigError as e:
        # Handle configuration errors during command execution
        logger.error(f"\n❌ {e.__class__.__name__}: {e.message}")
        if hasattr(e, "suggestion") and e.suggestion:
            logger.error(f"💡 Suggestion: {e.suggestion}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    # cli_app()
    main()
