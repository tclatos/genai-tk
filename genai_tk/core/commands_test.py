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

        @cli_app.command("fast_integration")
        def fast_integration(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose pytest output")] = False,
        ) -> None:
            """Run integration tests that do NOT require an LLM or API keys.

            Tests marked ``real_models`` or ``docker`` are automatically skipped.
            Safe to run in CI without any credentials.

            Examples:
                cli test fast_integration
                cli test fast_integration -v
            """
            import subprocess

            args = ["uv", "run", "pytest", "tests/integration_tests/"]
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

            Examples:
                cli test full_integration
                cli test full_integration --docker
                cli test full_integration --timeout 240 -v
            """
            import subprocess

            args = [
                "uv",
                "run",
                "pytest",
                "tests/integration_tests/",
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

            Examples:
                cli test all
                cli test all --docker --timeout 240
            """
            import subprocess

            args = [
                "uv",
                "run",
                "pytest",
                "tests/unit_tests/",
                "tests/integration_tests/",
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

            args = ["uv", "run", "pytest", "tests/", "-k", k_expr]
            if verbose:
                args.append("-v")
            if real:
                args.append("--include-real-models")
            result = subprocess.run(args)
            raise typer.Exit(result.returncode)
