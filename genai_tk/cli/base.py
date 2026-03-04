"""Base class for CLI command groups."""

from abc import ABC, abstractmethod

import typer
from pydantic import BaseModel


class CliTopCommand(BaseModel, ABC):
    """Base class for creating CLI command groups.

    This class provides a structured way to define command groups in the CLI.
    Subclasses must implement `get_description()` and `register_sub_commands()`.

    Example:
        ```python
        from typing import Annotated
        import typer
        from genai_tk.cli.base import CliTopCommand


        class DataCommands(CliTopCommand):
            '''Commands for data processing.'''

            description: str = "Data processing and transformation commands"

            def get_description(self) -> tuple[str, str]:
                return "data", self.description

            def register_sub_commands(self, cli_app: typer.Typer) -> None:
                @cli_app.command()
                def transform(input_file: Annotated[str, typer.Argument(help="Input file path")]) -> None:
                    '''Transform data from input file.'''
                    print(f"Transforming {input_file}...")
        ```

        Register in config:
        ```yaml
        cli:
          commands:
            - myapp.commands:DataCommands
        ```
    """

    def register(self, cli_app: typer.Typer) -> None:
        """Register this command group with the CLI application.

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
            A tuple of (command_name, description) where command_name is the CLI
            subcommand (e.g. ``"data"``) and description is the help text.

        Example:
            ```python
            def get_description(self) -> tuple[str, str]:
                return "database", "Database management commands"
            ```
        """
        ...
