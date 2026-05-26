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


def _copy_makefile(force: bool, app_name: str) -> None:
    """No-op — projects now use justfile instead of Makefile."""


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


def _pick_template() -> str:
    """Interactive template picker using questionary (falls back to typer.prompt)."""
    choices = [
        ("agent-app", "Agent App      — tools, skills, agent profiles, webapp page"),
        ("rag-app", "RAG Pipeline   — document ingestion, vector store, retrieval"),
        ("workflow-app", "Workflow App   — multi-step YAML pipelines, Prefect orchestration"),
        ("minimal", "Minimal        — config + justfile only, no example code"),
    ]
    try:
        import questionary
        from questionary import Choice

        result = questionary.select(
            "Choose a project template:",
            choices=[Choice(title=f"[{k}]  {v}", value=k) for k, v in choices],
        ).ask()
        return result or "agent-app"
    except ImportError:
        console.print("[bold]Available templates:[/bold]")
        for key, desc in choices:
            console.print(f"  [cyan]{key}[/cyan]  {desc}")
        console.print()
        chosen = typer.prompt(
            "Template",
            default="agent-app",
        )
        valid = {k for k, _ in choices}
        return chosen if chosen in valid else "agent-app"


def _print_next_steps(app_name: str, template: str, df_root: Path | None) -> None:
    """Print the post-init 'next steps' banner."""
    from genai_tk.main.scaffolder import TEMPLATE_META, _sanitize_package_name

    pkg = _sanitize_package_name(app_name)
    meta = TEMPLATE_META.get(template, {})
    run_cmd = meta.get("run_command", "cli --help")

    console.print("\n[bold green]✓ Done![/bold green]\n")
    console.print(f"  Project:  [bold]{app_name}[/bold]  (template: [cyan]{template}[/cyan])")
    if template != "minimal":
        console.print(f"  Package:  [bold]{pkg}/[/bold]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  [dim]uv sync[/dim]                                  install dependencies")
    console.print(f"  [dim]{run_cmd}[/dim]")
    console.print("  [dim]just skills[/dim]                              list available skills")
    console.print("  [dim]just lint[/dim]                                format + validate skills")

    console.print("\n[bold]Add community skills:[/bold]")
    console.print("  [cyan]cli skills add --skillssh langchain-ai/langchain-skills[/cyan]  [dim](LangChain)[/dim]")
    console.print("  [dim]Browse: https://www.skills.sh · npx skills add <owner/repo>[/dim]")

    if df_root:
        console.print("\n[bold]Deer-flow:[/bold] add to [bold].env[/bold]:")
        console.print(f"  [bold cyan]DEER_FLOW_PATH={df_root.absolute()}[/bold cyan]")

    console.print("\n[dim]Docs: AGENTS.md · docs/SKILLS.md · docs/EXTENDING.md[/dim]\n")


def _scaffold_project(
    project_dir: Path, project_name: str, *, template: str = "agent-app", force: bool = False
) -> None:
    """Generate Python package, examples, skills dir, and agent-support files."""
    try:
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, project_name, template=template, force=force)  # type: ignore[arg-type]
        scaffolder.scaffold()
    except ImportError:
        console.print("[yellow]Jinja2 not installed — skipping project scaffold.[/yellow]")
        console.print("  Install with: [bold]uv add jinja2[/bold]  then re-run [bold]cli init[/bold]\n")
    except Exception as exc:
        console.print(f"[red]Scaffold error:[/red] {exc}")


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
                typer.Option("--force", "-f", help="Overwrite existing files."),
            ] = False,
            name: Annotated[
                Optional[str],
                typer.Option("--name", "-n", help="Project name (default: current directory name)."),
            ] = None,
            template: Annotated[
                Optional[str],
                typer.Option(
                    "--template",
                    "-t",
                    help="Project template: agent-app | rag-app | workflow-app | minimal",
                ),
            ] = None,
            deer_flow: Annotated[
                bool,
                typer.Option("--deer-flow", "-d", help="Also clone and install the Deer-flow backend."),
            ] = False,
            deer_flow_path: Annotated[
                Optional[str],
                typer.Option("--deer-flow-path", help="Custom Deer-flow clone directory (default: ~/deer-flow)."),
            ] = None,
        ) -> None:
            """Initialize a new genai-tk project.

            \\b
            Examples:
                cli init                                   # interactive template picker
                cli init -t agent-app --name "My Project"  # agent app
                cli init -t rag-app                        # RAG pipeline
                cli init -t workflow-app                   # workflow orchestration
                cli init -t minimal                        # config + justfile only
                cli init --force                           # overwrite existing files
                cli init --deer-flow                       # also install Deer-flow
            """
            if ctx.invoked_subcommand is not None:
                return

            app_name = name or Path.cwd().name
            dest = Path.cwd() / "config"
            console.print(f"\n[bold]Initializing genai-tk project in[/bold] {Path.cwd()}\n")

            # ── Template selection ──────────────────────────────────────
            chosen_template = template
            if not chosen_template:
                chosen_template = _pick_template()

            # ── Config ─────────────────────────────────────────────────
            _copy_default_config(dest, force=force)
            _patch_webapp_yaml(dest, app_name)

            # ── Scaffold ────────────────────────────────────────────────
            if chosen_template != "minimal":
                _scaffold_project(Path.cwd(), app_name, template=chosen_template, force=force)
            else:
                # Minimal: just render common templates (no Python package)
                _scaffold_project(Path.cwd(), app_name, template="minimal", force=force)

            # ── Deer-flow (optional add-on) ──────────────────────────
            if deer_flow:
                df_path = Path(deer_flow_path) if deer_flow_path else None
                df_root = _install_deer_flow(df_path)
            else:
                df_root = None

            # ── Post-init banner ─────────────────────────────────────
            _print_next_steps(app_name, chosen_template, df_root)
