"""Utilities for displaying CLI command tree structure.

Provides functions to build and display a hierarchical tree view of Typer CLI commands.
"""

import typer
from rich.console import Console
from rich.tree import Tree


def build_command_tree(app: typer.Typer, parent_node: Tree | None = None, title: str | None = None) -> Tree:
    """Recursively build a tree of commands from a Typer app.

    Args:
        app: The Typer app to extract commands from
        parent_node: Parent tree node for nested commands
        title: Title for the root tree node

    Returns:
        Tree object containing the command structure
    """
    if parent_node is None:
        root = Tree(
            title or "[bold cyan]📋 Available Commands[/bold cyan]",
            guide_style="blue",
        )
    else:
        root = parent_node

    # Collect commands and groups separately
    commands = []
    groups = []

    # Get registered commands
    if hasattr(app, "registered_commands"):
        for cmd_info in app.registered_commands:
            cmd_name = cmd_info.name or cmd_info.callback.__name__
            cmd_help = (cmd_info.help or "").split("\n")[0]  # First line of help
            if not cmd_help and cmd_info.callback:
                # Try to get from docstring
                doc = cmd_info.callback.__doc__
                if doc:
                    cmd_help = doc.strip().split("\n")[0]
            commands.append((cmd_name, cmd_help))

    # Get registered groups (sub-apps)
    if hasattr(app, "registered_groups"):
        for group_info in app.registered_groups:
            group_name = group_info.name or ""
            # Extract help from the group's TyperInfo or from the typer instance itself
            group_help = None
            if hasattr(group_info, "help") and group_info.help:
                group_help = group_info.help
            elif group_info.typer_instance and hasattr(group_info.typer_instance, "info"):
                if hasattr(group_info.typer_instance.info, "help"):
                    group_help = group_info.typer_instance.info.help

            if not group_help:
                group_help = "Command group"

            groups.append((group_name, group_help, group_info.typer_instance))

    # Display commands first
    for cmd_name, cmd_help in sorted(commands):
        if cmd_help:
            root.add(f"[green]{cmd_name}[/green]: [dim]{cmd_help}[/dim]")
        else:
            root.add(f"[green]{cmd_name}[/green]")

    # Then display groups with their subcommands
    for group_name, group_help, typer_instance in sorted(groups, key=lambda x: x[0]):
        # Create group node with description
        group_node = root.add(f"[bold yellow]{group_name}[/bold yellow]: [italic dim]{group_help}[/italic dim]")

        # Recursively add subcommands to the group
        if typer_instance:
            build_command_tree(typer_instance, group_node)

    return root


def display_command_tree(
    app: typer.Typer,
    title: str | None = None,
    show_usage_examples: bool = True,
    console: Console | None = None,
) -> None:
    """Display a tree view of all CLI commands.

    Args:
        app: The Typer app to display commands from
        title: Custom title for the command tree
        show_usage_examples: Whether to show usage examples
        console: Rich console instance (creates new one if not provided)
    """
    if console is None:
        console = Console()

    # Build and display the tree
    tree = build_command_tree(app, title=title)
    console.print(tree)

    # Add usage examples
    if show_usage_examples:
        console.print("\n[bold cyan]💡 Quick Start:[/bold cyan]")
        console.print("  [dim]$[/dim] uv run cli [bold yellow]<group>[/bold yellow] [bold green]<command>[/bold green] [dim]# Run a command[/dim]")
        console.print("  [dim]$[/dim] uv run cli [bold yellow]<group>[/bold yellow] --help         [dim]# Show help for a group[/dim]")
        console.print("  [dim]$[/dim] uv run cli info config              [dim]# Example: show configuration[/dim]")
        console.print("\n[dim]For detailed help on any command, use: [/dim][bold]uv run cli <group> <command> --help[/bold]\n")
