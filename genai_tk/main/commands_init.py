"""Bootstrap CLI commands — available even without a config file.

Registered statically in cli.py so they work from an empty directory
(i.e. immediately after `uv add genai-tk`).

Usage:
    cli init                    # copy default config to ./config/
    cli init --force            # overwrite existing config files
    cli init --deer-flow        # also install the Deer-flow backend
    cli init --deer-flow --path ~/my-deer-flow
"""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from genai_tk.cli.base import CliTopCommand

console = Console()

_CONFIG_PKG_PATH = "default_config"  # inside genai_tk package (wheel-bundled)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_default_config(dest: Path, force: bool) -> bool:
    """Copy the bundled default config to *dest*.

    Returns True if any files were written.
    """
    try:
        src_root = pkg_files("genai_tk") / _CONFIG_PKG_PATH
    except Exception as exc:
        console.print(f"[red]Could not locate bundled config:[/red] {exc}")
        return False

    written = 0
    skipped = 0
    for item in src_root.iterdir():
        _copy_tree(item, dest / item.name, force=force, written_ref=[0, 0])

    # Re-do as a proper recursive walk with counters
    written, skipped = _copy_tree_counted(src_root, dest, force=force)

    if written:
        console.print(f"[green]✓ Copied {written} config file(s) to[/green] [bold]{dest}[/bold]")
    else:
        console.print(f"[yellow]No files written ({skipped} already exist — use --force to overwrite)[/yellow]")
    return written > 0


def _copy_tree_counted(src_traversable, dest: Path, force: bool) -> tuple[int, int]:
    """Recursively copy a Traversable tree to dest, returns (written, skipped)."""
    written = skipped = 0
    dest.mkdir(parents=True, exist_ok=True)
    for item in src_traversable.iterdir():
        target = dest / item.name
        if item.is_dir():
            w, s = _copy_tree_counted(item, target, force=force)
            written += w
            skipped += s
        else:
            if target.exists() and not force:
                skipped += 1
            else:
                target.write_bytes(item.read_bytes())
                written += 1
    return written, skipped


def _copy_tree(src_traversable, dest: Path, force: bool, written_ref) -> None:
    """Noop — only _copy_tree_counted is used."""


def _install_deer_flow(path: Path | None) -> bool:
    """Clone/update Deer-flow and install its backend. Returns True on success."""
    deer_flow_repo = "https://github.com/bytedance/deer-flow.git"
    target = path.expanduser().resolve() if path else Path.home() / "deer-flow"

    if target.exists():
        console.print(f"[cyan]Updating Deer-flow at {target} ...[/cyan]")
        result = subprocess.run(["git", "-C", str(target), "pull", "--rebase"], capture_output=True, text=True)
    else:
        console.print(f"[cyan]Cloning Deer-flow into {target} ...[/cyan]")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", deer_flow_repo, str(target)], capture_output=True, text=True
        )
    if result.returncode != 0:
        console.print(f"[red]git failed:[/red] {result.stderr.strip()}")
        return False

    backend = target / "backend"
    if not backend.exists():
        console.print(f"[red]Backend directory not found:[/red] {backend}")
        return False

    console.print(f"[cyan]Installing Deer-flow backend ...[/cyan]")
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(backend)], capture_output=True, text=True
    )
    if install.returncode != 0:
        console.print(f"[red]Install failed:[/red] {install.stderr.strip()}")
        return False

    console.print(f"[green]✓ Deer-flow installed.[/green]")
    console.print(f"\nAdd to your [bold].env[/bold]:\n  [bold cyan]DEER_FLOW_PATH={target}[/bold cyan]\n")
    return True


# ---------------------------------------------------------------------------
# Command class
# ---------------------------------------------------------------------------


class InitCommands(CliTopCommand):
    """Bootstrap project initialisation — no config required."""

    description: str = "Initialize a new project with default configuration"

    def get_description(self) -> tuple[str, str]:
        return "init", self.description

    def register(self, cli_app: typer.Typer) -> None:
        # Override base: disable no_args_is_help so the callback runs with zero args.
        command_name, description = self.get_description()
        sub_app = typer.Typer(no_args_is_help=False, help=description)
        self.register_sub_commands(sub_app)
        cli_app.add_typer(sub_app, name=command_name)

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.callback(invoke_without_command=True)
        def init(
            ctx: typer.Context,
            force: Annotated[
                bool,
                typer.Option("--force", "-f", help="Overwrite existing config files."),
            ] = False,
            deer_flow: Annotated[
                bool,
                typer.Option("--deer-flow", "-d", help="Also clone and install the Deer-flow backend."),
            ] = False,
            deer_flow_path: Annotated[
                Optional[str],
                typer.Option(
                    "--deer-flow-path",
                    help="Custom Deer-flow clone directory (default: ~/deer-flow).",
                ),
            ] = None,
        ) -> None:
            """Copy default config files to ./config/ and optionally install Deer-flow.

            Run this once in a new project after installing genai-tk:

            \\b
            Examples:
                cli init
                cli init --force                           # overwrite existing files
                cli init --deer-flow                       # also install Deer-flow
                cli init --deer-flow --deer-flow-path ~/projects/deer-flow
            """
            if ctx.invoked_subcommand is not None:
                return

            dest = Path.cwd() / "config"
            console.print(f"\n[bold]Initializing genai-tk project in[/bold] {Path.cwd()}\n")

            _copy_default_config(dest, force=force)

            if deer_flow:
                df_path = Path(deer_flow_path) if deer_flow_path else None
                _install_deer_flow(df_path)

            console.print("\n[bold green]Done.[/bold green] You can now run [bold]cli[/bold] commands.\n")
            if not deer_flow:
                console.print(
                    "Tip: run [bold]cli init --deer-flow[/bold] to also install the Deer-flow backend.\n"
                )
