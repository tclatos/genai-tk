"""Bootstrap CLI commands — available even without a config file.

Registered statically in cli.py so they work from an empty directory
(i.e. immediately after `uv add genai-tk`).

Usage:
    cli init                                    # interactive template picker
    cli init -t agent-app --name "My Project"   # agent app scaffold
    cli init -t rag-app                         # RAG pipeline scaffold
    cli init -t workflow-app                    # workflow scaffold
    cli init -t minimal                         # config + justfile only
    cli init --force                            # overwrite existing files
    cli init --deer-flow                        # also install Deer-flow
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


def _patch_webapp_yaml(config_dir: Path, app_name: str) -> None:
    """Replace the placeholder app_name in the copied webapp.yaml."""
    webapp_yaml = config_dir / "webapp.yaml"
    if not webapp_yaml.exists():
        return
    content = webapp_yaml.read_text(encoding="utf-8")
    patched = content.replace("app_name: GenAI Toolkit", f"app_name: {app_name}")
    if patched == content:
        # Fallback for legacy template
        patched = content.replace("app_name: My GenAI Project", f"app_name: {app_name}")
    if patched != content:
        webapp_yaml.write_text(patched, encoding="utf-8")
        console.print(f"[green]✓ app_name set to[/green] [bold]{app_name!r}[/bold] in {webapp_yaml}")


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


_SCAFFOLD_EXCLUDE = {".template"}  # file suffixes never copied to user projects


def _copy_tree_counted(src_traversable, dest: Path, force: bool) -> tuple[int, int]:
    """Recursively copy a Traversable tree to dest, returns (written, skipped)."""
    written = skipped = 0
    dest.mkdir(parents=True, exist_ok=True)
    for item in src_traversable.iterdir():
        if Path(item.name).suffix in _SCAFFOLD_EXCLUDE:
            continue
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


def _install_deer_flow_package() -> bool:
    """Install deerflow-harness from GitHub as a proper uv dependency."""
    spec = "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
    console.print("[cyan]Installing deerflow-harness via uv ...[/cyan]")
    result = subprocess.run(["uv", "add", spec], capture_output=False, text=True)
    if result.returncode != 0:
        console.print("[red]Failed to install deerflow-harness.[/red]")
        return False
    console.print("[green]✓ deerflow-harness installed.[/green]")
    return True


def _install_sandbox_packages() -> bool:
    """Install aio-sandbox optional dependency group (agent-sandbox + opensandbox)."""
    console.print("[cyan]Installing aio-sandbox packages via uv ...[/cyan]")
    result = subprocess.run(["uv", "sync", "--group", "aio-sandbox"], capture_output=False, text=True)
    if result.returncode != 0:
        console.print("[red]Failed to install aio-sandbox packages.[/red]")
        return False
    console.print("[green]\u2713 aio-sandbox installed (agent-sandbox + opensandbox + opensandbox-server).[/green]")
    return True


def _print_next_steps(app_name: str, deer_flow: bool, sandbox: bool) -> None:
    """Print the post-init 'next steps' banner."""
    from genai_tk.main.scaffolder import _sanitize_package_name

    pkg = _sanitize_package_name(app_name)

    console.print("\n[bold green]✓ Done![/bold green]\n")
    console.print(f"  Project:  [bold]{app_name}[/bold]")
    console.print(f"  Package:  [bold]{pkg}/[/bold]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  [dim]uv sync[/dim]                    install dependencies")
    console.print("  [dim]uv run cli agent chat[/dim]      start agent chat")
    console.print("  [dim]just skills[/dim]               list available skills")

    console.print("\n[bold]IDE setup[/bold] [dim](copy the command for your IDE):[/dim]")
    console.print("  VS Code + Copilot : already configured via .github/copilot-instructions.md")
    console.print("  Cursor            : [dim]cp AGENTS.md .cursor/rules/project.md[/dim]")
    console.print("  Windsurf          : [dim]cp AGENTS.md .windsurfrules[/dim]")
    console.print("  Claude Code       : [dim]cp AGENTS.md CLAUDE.md[/dim]")

    console.print("\n[bold]Install ecosystem skills for your IDE:[/bold]")
    console.print("  [cyan]cli skills add --skillssh langchain-ai/langchain-skills[/cyan]  [dim](LangChain)[/dim]")
    console.print("  [dim]Browse: https://www.skills.sh · npx skills add <owner/repo>[/dim]")

    if deer_flow:
        console.print("\n[bold]Deer-flow:[/bold] installed as [cyan]deerflow-harness[/cyan] in your venv.")
        console.print("  To update: [dim]uv add deerflow-harness @ git+...@main#subdirectory=...[/dim]")
    if sandbox:
        console.print("\n[bold]AIO Sandbox:[/bold] installed ([cyan]agent-sandbox + opensandbox[/cyan]).")
        console.print("  Start server: [dim]opensandbox-server start[/dim]")
        console.print("  Then set sandbox: [dim]type: aio_sandbox[/dim] in your agent profile YAML.")
    console.print("\n[dim]Docs: AGENTS.md · docs/SKILLS.md · docs/EXTENDING.md[/dim]\n")


def _scaffold_project(project_dir: Path, project_name: str, *, force: bool = False) -> None:
    """Generate Python package, examples, skills dir, and agent-support files."""
    try:
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, project_name, force=force)
        scaffolder.scaffold()
    except ImportError:
        console.print("[yellow]Jinja2 not installed — skipping project scaffold.[/yellow]")
        console.print("  Install with: [bold]uv add jinja2[/bold]  then re-run [bold]cli init[/bold]\n")
    except Exception as exc:
        console.print(f"[red]Scaffold error:[/red] {exc}")


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
                typer.Option("--force", "-f", help="Overwrite existing files."),
            ] = False,
            name: Annotated[
                Optional[str],
                typer.Option("--name", "-n", help="Project name (default: current directory name)."),
            ] = None,
            with_deer_flow: Annotated[
                bool,
                typer.Option("--with-deer-flow", help="Install deerflow-harness (heavy optional component)."),
            ] = False,
            with_sandbox: Annotated[
                bool,
                typer.Option("--with-sandbox", help="Install aio-sandbox packages (agent-sandbox + opensandbox)."),
            ] = False,
        ) -> None:
            """Initialize a new genai-tk project.

            \\b
            Examples:
                cli init                           # scaffold project in current directory
                cli init --name "My Project"        # set project name
                cli init --force                   # overwrite existing files
                cli init --with-deer-flow          # also install Deer-flow
                cli init --with-sandbox            # also install aio-sandbox
            """
            if ctx.invoked_subcommand is not None:
                return

            app_name = name or Path.cwd().name
            dest = Path.cwd() / "config"
            console.print(f"\n[bold]Initializing genai-tk project in[/bold] {Path.cwd()}\n")

            # ── Config ─────────────────────────────────────────────────
            _copy_default_config(dest, force=force)
            _patch_webapp_yaml(dest, app_name)

            # ── Scaffold ────────────────────────────────────────────────
            _scaffold_project(Path.cwd(), app_name, force=force)

            # ── Deer-flow (optional add-on) ──────────────────────────
            if with_deer_flow:
                _install_deer_flow_package()
            if with_sandbox:
                _install_sandbox_packages()

            # ── Post-init banner ─────────────────────────────────────
            _print_next_steps(app_name, deer_flow=with_deer_flow, sandbox=with_sandbox)
