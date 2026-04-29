"""CLI commands for running test suites via pytest and notebook execution.

Provides a `test` command group so tests can be invoked uniformly through the
CLI across all projects that vendor genai-tk.

Test paths are resolved from the ``test`` section of ``app_conf.yaml``; if a
key is absent a warning is printed and common fallback paths are tried.

Note: the pytest-based sub-commands are thin wrappers — if the package itself
is broken, fall back to ``uv run pytest`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from genai_tk.cli.base import CliTopCommand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FALLBACKS: dict[str, list[str]] = {
    "unit": ["tests/unit_tests", "tests/unit", "tests"],
    "integration": ["tests/integration_tests", "tests/integration", "tests"],
    "evals": ["tests/eval_tests", "tests/evals", "tests"],
    "notebooks": ["notebooks"],
}


def _resolve_test_path(key: str) -> Path:
    """Return the configured path for *key*, falling back to common defaults.

    Emits a Rich warning when the config key is absent so the user knows to
    add ``test.<key>`` in ``app_conf.yaml``.
    """
    from rich.console import Console

    console = Console(stderr=True)

    try:
        from genai_tk.utils.config_mngr import global_config

        cfg_path = global_config().get_str(f"test.{key}", default=None)
    except Exception:  # noqa: BLE001 — config unavailable during bootstrap
        cfg_path = None

    if cfg_path:
        return Path(cfg_path)

    console.print(
        f"[yellow]warning:[/yellow] [bold]test.{key}[/bold] not set in app_conf.yaml — falling back to common paths."
    )
    for candidate in _FALLBACKS.get(key, []):
        p = Path(candidate)
        if p.exists():
            console.print(f"  [dim]using fallback:[/dim] {p}")
            return p

    # Return the first fallback even if it doesn't exist; pytest will report clearly.
    return Path(_FALLBACKS[key][0])


class TestCommands(CliTopCommand):
    """Commands for running test suites."""

    description: str = "Run test suites via pytest."

    def get_description(self) -> tuple[str, str]:
        return "test", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("unit")
        def unit(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            """Run unit tests only (configured via test.unit in app_conf.yaml)."""
            import subprocess

            path = _resolve_test_path("unit")
            args = ["uv", "run", "pytest", str(path)]
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("fast_integration")
        def fast_integration(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            """Run integration tests that do NOT require an LLM or API keys.

            Tests marked ``real_models`` or ``docker`` are automatically skipped.
            Safe to run in CI without any credentials.
            Path is configured via ``test.integration`` in app_conf.yaml.

            Examples:
                cli test fast_integration
                cli test fast_integration -v
            """
            import subprocess

            path = _resolve_test_path("integration")
            args = ["uv", "run", "pytest", str(path)]
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("full_integration")
        def full_integration(
            docker: Annotated[bool, typer.Option("--docker", help="Include tests that require Docker")] = False,
            timeout: Annotated[int, typer.Option("--timeout", help="Per-test timeout in seconds")] = 120,
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            """Run integration tests including those that require real LLM API keys.

            Passes ``--include-real-models`` to pytest so that tests marked
            ``real_models`` are executed.  DeerFlow tests also run when
            ``DEER_FLOW_PATH`` is set in the environment.
            Path is configured via ``test.integration`` in app_conf.yaml.

            Examples:
                cli test full_integration
                cli test full_integration --docker
                cli test full_integration --timeout 240 -v
            """
            import subprocess

            path = _resolve_test_path("integration")
            args = [
                "uv",
                "run",
                "pytest",
                str(path),
                "--include-real-models",
                f"--timeout={timeout}",
            ]
            if docker:
                args.append("--include-docker")
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("all")
        def all_(
            docker: Annotated[bool, typer.Option("--docker", help="Include tests that require Docker")] = False,
            timeout: Annotated[int, typer.Option("--timeout", help="Per-test timeout in seconds")] = 120,
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            """Run unit tests + full integration tests (real LLM calls included).

            Equivalent to running ``unit`` followed by ``full_integration``.
            Evals are excluded — use ``cli test evals`` for those.
            Paths are configured via ``test.unit`` / ``test.integration`` in app_conf.yaml.

            Examples:
                cli test all
                cli test all --docker --timeout 240
            """
            import subprocess

            unit_path = _resolve_test_path("unit")
            integ_path = _resolve_test_path("integration")
            args = [
                "uv",
                "run",
                "pytest",
                str(unit_path),
                str(integ_path),
                "--include-real-models",
                "-m",
                "not slow and not evals",
                f"--timeout={timeout}",
            ]
            if docker:
                args.append("--include-docker")
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("evals")
        def evals(
            real: Annotated[bool, typer.Option("--real", help="Include LLM-judged tests (requires API key)")] = False,
            deerflow: Annotated[
                bool, typer.Option("--deerflow", help="Run deerflow eval tests (requires DEER_FLOW_PATH + API key)")
            ] = False,
            timeout: Annotated[int, typer.Option("--timeout", help="Per-test timeout in seconds")] = 60,
        ) -> None:
            """Run eval tests.

            By default runs only deterministic evals (no API key needed).
            Use --real to include LLM-judged tests, --deerflow for the deerflow suite.
            Path is configured via ``test.evals`` in app_conf.yaml.

            Examples:
                uv run cli test evals
                uv run cli test evals --real
                uv run cli test evals --deerflow --timeout 360
            """
            import subprocess

            path = _resolve_test_path("evals")
            if deerflow:
                marker = "evals and deerflow"
                extra = ["--include-real-models", f"--timeout={timeout}"]
            elif real:
                marker = "evals"
                extra = ["--include-real-models", f"--timeout={timeout}"]
            else:
                marker = "evals and not real_models"
                extra = []

            args = ["uv", "run", "pytest", str(path), "-m", marker, "-v"] + extra
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("select")
        def select(
            pattern: Annotated[str, typer.Argument(help="Pattern to match test names (e.g. '*deerflow*')")],
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
            real: Annotated[bool, typer.Option("--real", help="Include tests that require real LLM API keys")] = False,
        ) -> None:
            """Run tests whose name matches a pattern across all test directories.

            The pattern is matched against test names using pytest's ``-k`` option
            (substring/expression matching).  Glob wildcards (``*``) are stripped
            automatically since pytest performs substring matching by default.

            Examples:
                cli test select '*deerflow*'
                cli test select 'rag' -v
                cli test select 'embedding or vectorstore' --real
            """
            import subprocess

            # Normalise glob wildcards: pytest -k does substring matching, so '*foo*' == 'foo'
            k_expr = pattern.replace("*", "").strip()
            if not k_expr:
                import typer as _typer

                _typer.echo("Pattern must contain at least one non-wildcard character.", err=True)
                raise typer.Exit(1)

            # Collect all configured test directories as search roots.
            seen: set[Path] = set()
            roots: list[str] = []
            for key in ("unit", "integration", "evals"):
                p = _resolve_test_path(key)
                if p not in seen:
                    seen.add(p)
                    roots.append(str(p))

            args = ["uv", "run", "pytest"] + roots + ["-k", k_expr]
            if verbose:
                args.append("-v")
            if real:
                args.append("--include-real-models")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("notebooks")
        def notebooks(
            path: Annotated[
                str | None,
                typer.Argument(help="Path to a single .ipynb file or a directory. Defaults to test.notebooks config."),
            ] = None,
            allow_pip: Annotated[
                bool, typer.Option("--allow-pip", help="Execute cells that contain %pip / !pip commands")
            ] = False,
            glob: Annotated[
                str, typer.Option("--glob", help="Glob pattern for discovering notebooks in a directory")
            ] = "**/*.ipynb",
            quiet: Annotated[
                bool, typer.Option("--quiet", "-q", help="Suppress execution output; only show summary table")
            ] = False,
        ) -> None:
            """Execute Jupyter notebooks and report pass/fail for each.

            Runs every code cell in order using a shared exec() namespace (the
            lightweight approach — no Jupyter kernel required).  Stops at the
            first failing cell per notebook.

            Path is configured via ``test.notebooks`` in app_conf.yaml.

            Examples:
                cli test notebooks
                cli test notebooks notebooks/my_demo.ipynb
                cli test notebooks --glob "*.ipynb"
                cli test notebooks --allow-pip
                cli test notebooks --quiet
            """
            from rich import box
            from rich.console import Console
            from rich.table import Table

            from genai_tk.utils.notebook_runner import run_notebook

            console = Console()

            # Resolve target path
            target = Path(path) if path else _resolve_test_path("notebooks")

            if target.is_file():
                nb_files = [target]
            else:
                nb_files = sorted(target.glob(glob))

            if not nb_files:
                console.print(f"[yellow]No notebooks found in[/yellow] {target}")
                raise typer.Exit(0)

            if not quiet:
                console.print(f"\n[bold]Running {len(nb_files)} notebook(s)[/bold] from [cyan]{target}[/cyan]\n")

            table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
            table.add_column("Notebook", style="cyan", no_wrap=True)
            table.add_column("Cells run", justify="right")
            table.add_column("Skipped", justify="right")
            table.add_column("Duration", justify="right")
            table.add_column("Status", justify="center")

            failed: list[str] = []

            for nb_path in nb_files:
                if not quiet:
                    console.print(f"  [dim]executing[/dim] {nb_path.name} ...", end="")
                result = run_notebook(nb_path, allow_pip=allow_pip, suppress_output=quiet)
                status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                if not quiet:
                    console.print(f"\r  {status} {nb_path.name}          ")

                if not result.passed:
                    failed.append(str(nb_path))
                    if not quiet:
                        for cell_res in result.failed_cells:
                            console.print(
                                f"    [red]Cell {cell_res.cell_index}:[/red] "
                                f"{type(cell_res.error).__name__}: {cell_res.error}"
                            )

                table.add_row(
                    str(nb_path.relative_to(Path.cwd()) if nb_path.is_absolute() else nb_path),
                    str(len(result.cell_results)),
                    str(result.skipped),
                    f"{result.total_duration:.2f}s",
                    status,
                )

            if not quiet:
                console.print()
            console.print(table)

            if failed:
                console.print(f"\n[red bold]{len(failed)} notebook(s) FAILED[/red bold]")
                raise typer.Exit(1)
            else:
                console.print(f"\n[green bold]All {len(nb_files)} notebook(s) passed[/green bold]")
                raise typer.Exit(0)
