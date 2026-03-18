"""Unit tests for genai_tk.cli.command_tree."""

import typer
from rich.console import Console
from rich.tree import Tree

from genai_tk.cli.command_tree import build_command_tree, display_command_tree


def _make_app() -> typer.Typer:
    """Create a simple test Typer app."""
    app = typer.Typer(no_args_is_help=True)

    @app.command()
    def hello(name: str = "world") -> None:
        """Say hello."""
        print(f"Hello, {name}!")

    @app.command()
    def goodbye() -> None:
        """Say goodbye."""
        print("Goodbye!")

    return app


def _make_nested_app() -> typer.Typer:
    """Create a nested Typer app with subcommands."""
    app = typer.Typer(no_args_is_help=True)
    sub = typer.Typer(help="Sub group commands")

    @sub.command()
    def sub_cmd() -> None:
        """Sub command."""

    app.add_typer(sub, name="subgroup")
    return app


class TestBuildCommandTree:
    def test_returns_tree_object(self) -> None:
        app = _make_app()
        result = build_command_tree(app)
        assert isinstance(result, Tree)

    def test_includes_commands(self) -> None:
        app = _make_app()
        tree = build_command_tree(app)
        tree_str = str(tree)
        assert isinstance(tree_str, str)

    def test_custom_title(self) -> None:
        app = _make_app()
        tree = build_command_tree(app, title="Custom Title")
        assert isinstance(tree, Tree)

    def test_with_parent_node(self) -> None:
        app = _make_app()
        parent = Tree("Root")
        result = build_command_tree(app, parent_node=parent)
        assert result is parent  # Returns parent when provided

    def test_nested_app(self) -> None:
        app = _make_nested_app()
        tree = build_command_tree(app)
        assert isinstance(tree, Tree)

    def test_empty_app(self) -> None:
        app = typer.Typer()
        tree = build_command_tree(app)
        assert isinstance(tree, Tree)


class TestDisplayCommandTree:
    def test_displays_without_error(self) -> None:
        app = _make_app()
        console = Console(force_terminal=False)
        display_command_tree(app, console=console)

    def test_shows_usage_examples_by_default(self) -> None:
        from io import StringIO

        app = _make_app()
        output = StringIO()
        console = Console(file=output, force_terminal=False)
        display_command_tree(app, console=console, show_usage_examples=True)
        text = output.getvalue()
        assert "Quick Start" in text or len(text) > 0

    def test_no_usage_examples(self) -> None:
        from io import StringIO

        app = _make_app()
        output = StringIO()
        console = Console(file=output, force_terminal=False)
        display_command_tree(app, console=console, show_usage_examples=False)
        text = output.getvalue()
        assert "Quick Start" not in text

    def test_custom_title(self) -> None:
        from io import StringIO

        app = _make_app()
        output = StringIO()
        console = Console(file=output, force_terminal=False)
        display_command_tree(app, title="Test Commands", console=console)

    def test_creates_console_if_none(self) -> None:
        app = _make_app()
        # Should not raise even without providing console
        display_command_tree(app)
