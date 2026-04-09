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


def _install_deer_flow(path: Path | None) -> Path | None:
    """Clone/update Deer-flow and install its backend. Returns root path on success, None on failure."""
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
        return None

    backend = target / "backend"
    if not backend.exists():
        console.print(f"[red]Backend directory not found:[/red] {backend}")
        return None

    console.print("[cyan]Installing Deer-flow backend dependencies ...[/cyan]")
    if not _install_deer_flow_backend(backend):
        return None

    # Validate Deer-flow installation
    _validate_deer_flow_installation(target)

    console.print("[green]✓ Deer-flow installed.[/green]")
    return target


def _validate_deer_flow_installation(root: Path) -> None:
    """Validate Deer-flow installation and show warnings if issues are found."""
    import subprocess
    
    backend = root / "backend"
    
    # Check expected directories
    harness = backend / "packages" / "harness"
    legacy_src = backend / "src"
    
    if not harness.exists() and not legacy_src.exists():
        console.print(
            f"[yellow]Warning:[/yellow] Expected backend/packages/harness or backend/src not found.\n"
            f"  Deer-flow at {root} may be incomplete or using an unexpected layout."
        )
    
    # Check pyproject.toml exists
    if not (backend / "pyproject.toml").exists():
        console.print(
            f"[yellow]Warning:[/yellow] pyproject.toml not found in {backend}.\n"
            f"  Deer-flow installation may be incomplete."
        )
    
    # Check git version
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%h %ai %s"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            info = result.stdout.strip().split()
            if len(info) >= 3:
                commit_hash = info[0]
                date = info[1]
                console.print(f"[cyan]Deer-flow version:[/cyan] {commit_hash} ({date})")
        else:
            console.print("[yellow]Warning:[/yellow] Could not determine Deer-flow version from git.")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not check Deer-flow version: {e}")





def _install_deer_flow_backend(backend: Path) -> bool:
    """Install deer-flow's deps + harness package without building deer-flow itself.

    deer-flow's backend has no [build-system] and a flat layout that confuses
    setuptools.  The embedded client resolves the code via sys.path, so we only
    need the runtime dependencies and the harness sub-package.
    """
    import tomllib

    pyproject = backend / "pyproject.toml"
    if not pyproject.exists():
        console.print(f"[red]pyproject.toml not found:[/red] {pyproject}")
        return False

    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    deps: list[str] = data.get("project", {}).get("dependencies", [])

    # Install harness (it has a proper layout) + all runtime deps
    harness = backend / "packages" / "harness"
    cmd: list[str] = ["uv", "pip", "install"]
    if harness.exists():
        cmd += ["-e", str(harness)]
    cmd += deps

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Install failed:[/red] {result.stderr.strip()}")
        return False
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
                df_root = _install_deer_flow(df_path)
            else:
                df_root = None

            console.print("\n[bold green]Done.[/bold green]\n")
            
            if df_root:
                console.print(
                    f"[bold]Set up Deer-flow in your .env:[/bold]\n"
                    f"  [bold cyan]DEER_FLOW_PATH={df_root.absolute()}[/bold cyan]\n"
                )
            
            console.print("You can now run [bold]cli[/bold] commands.\n")
            if not deer_flow:
                console.print("Tip: run [bold]cli init --deer-flow[/bold] to also install the Deer-flow backend.\n")
