"""CLI commands for running test suites via pytest.

Provides a `test` command group so tests can be invoked uniformly through the
CLI across all projects that vendor genai-tk.

Note: this is intentionally a thin wrapper around pytest — no logic lives here.
If the package itself is broken, fall back to `uv run pytest` directly.
"""

from typing import Annotated

import typer

from genai_tk.cli.base import CliTopCommand


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
            """Run unit tests only (tests/unit_tests/)."""
            import subprocess

            args = ["uv", "run", "pytest", "tests/unit_tests/"]
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("integration")
        def integration(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            import subprocess

            """Run integration tests only (tests/integration_tests/)."""
            args = ["uv", "run", "pytest", "tests/integration_tests/"]
            if verbose:
                args.append("-v")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("all")
        def all_(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            import subprocess

            """Run unit + integration tests (default: no evals, no real models)."""
            args = ["uv", "run", "pytest", "tests/unit_tests/", "tests/integration_tests/"]
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

            Examples:
                uv run cli test evals
                uv run cli test evals --real
                uv run cli test evals --deerflow --timeout 360
            """
            import subprocess

            if deerflow:
                marker = "evals and deerflow"
                extra = ["--include-real-models", f"--timeout={timeout}"]
            elif real:
                marker = "evals"
                extra = ["--include-real-models", f"--timeout={timeout}"]
            else:
                marker = "evals and not real_models"
                extra = []

            args = ["uv", "run", "pytest", "tests/eval_tests/", "-m", marker, "-v"] + extra
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)

        @cli_app.command("full")
        def full(
            timeout: Annotated[int, typer.Option("--timeout", help="Per-test timeout in seconds")] = 120,
        ) -> None:
            """Run ALL tests including real LLM calls (requires API keys)."""
            import subprocess

            args = [
                "uv",
                "run",
                "pytest",
                "tests/unit_tests/",
                "tests/integration_tests/",
                "--include-real-models",
                "-m",
                "not slow",
                "-v",
                f"--timeout={timeout}",
            ]
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)
