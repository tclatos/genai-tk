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
from genai_tk.main.cli import CliTopCommand

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
from abc import ABC, abstractmethod

import typer
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

# Import modules where runnables are registered
from genai_tk.utils.cli.command_tree import display_command_tree
from genai_tk.utils.config_mngr import global_config, import_from_qualified
from genai_tk.utils.logger_factory import setup_logging

load_dotenv(verbose=True)


class CliTopCommand(BaseModel, ABC):
    """Base class for creating CLI command groups.

    This class provides a structured way to define command groups in the CLI.
    Subclasses must implement `get_description()` and `register_sub_commands()`.

    Attributes:
        description: A brief description of the command group shown in help text.

    Example:
        ```python
        from typing import Annotated
        import typer
        from genai_tk.main.cli import CliTopCommand

        class DataCommands(CliTopCommand):
            '''Commands for data processing.'''

            description: str = "Data processing and transformation commands"

            def get_description(self) -> tuple[str, str]:
                return "data", self.description

            def register_sub_commands(self, cli_app: typer.Typer) -> None:
                @cli_app.command()
                def transform(
                    input_file: Annotated[str, typer.Argument(help="Input file path")]
                ) -> None:
                    '''Transform data from input file.'''
                    print(f"Transforming {input_file}...")

                @cli_app.command()
                def validate(
                    schema: Annotated[str, typer.Option("--schema", help="Validation schema")]
                ) -> None:
                    '''Validate data against schema.'''
                    print(f"Validating with schema: {schema}")
        ```

        Register in config:
        ```yaml
        cli:
          commands:
            - myapp.commands:DataCommands
        ```

        Usage:
        ```bash
        uv run cli data transform input.json
        uv run cli data validate --schema schema.json
        uv run cli data --help
        ```
    """

    def register(self, cli_app: typer.Typer) -> None:
        """Register this command group with the CLI application.

        This method is called automatically by the CLI loader. It creates a
        sub-application for the command group and registers all sub-commands.

        Args:
            cli_app: The main Typer CLI application to register commands with.
        """
        command_name, description = self.get_description()
        sub_app = typer.Typer(no_args_is_help=True, help=description)
        self.register_sub_commands(sub_app)
        cli_app.add_typer(sub_app, name=command_name)

    @abstractmethod
    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        """Register individual commands for this command group.

        Implement this method to define the commands that belong to this group.
        Use the `@cli_app.command()` decorator to register each command.

        Args:
            cli_app: The Typer sub-application for this command group.

        Example:
            ```python
            def register_sub_commands(self, cli_app: typer.Typer) -> None:
                @cli_app.command()
                def list() -> None:
                    '''List all items.'''
                    print("Listing items...")

                @cli_app.command()
                def create(name: str) -> None:
                    '''Create a new item.'''
                    print(f"Creating {name}...")
            ```
        """
        ...

    @abstractmethod
    def get_description(self) -> tuple[str, str]:
        """Return the command group name and description.

        Returns:
            A tuple of (command_name, description) where:
            - command_name: The CLI command name (e.g., "data", "config")
            - description: A brief description shown in help text

        Example:
            ```python
            def get_description(self) -> tuple[str, str]:
                return "database", "Database management commands"
            ```

            This will create a command group accessible via:
            ```bash
            uv run cli database --help
            ```
        """
        ...


PRETTY_EXCEPTION = (
    False  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/
)

cli_app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=PRETTY_EXCEPTION,
)


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
    # We could fo better with Typer @cli_app.callback(), but I haven't succeded
    if "--logging" in sys.argv:
        level = "TRACE"
        sys.argv.remove("--logging")
    else:
        level = None

    setup_logging(level)
    modules = global_config().get_list("cli.commands", value_type=str)
    # Import and register commands from each module

    for module in modules:
        try:
            # debug(module)
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

    # Check if --help is requested or no arguments provided (show custom tree instead of default help)
    if len(sys.argv) == 1 or ("--help" in sys.argv and len(sys.argv) == 2):
        display_command_tree(cli_app)
        return

    cli_app()


if __name__ == "__main__":
    # cli_app()
    main()
