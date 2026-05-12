"""CLI commands for the workflow configuration layer."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genai_tk.cli.base import CliTopCommand
from genai_tk.workflow.resolver import WorkflowResolutionError, parse_cli_overrides, resolve_workflow_invocation


def _render_workflow_summary(resolved_name: str, invocation: object) -> None:
    from genai_tk.workflow.models import ResolvedWorkflowInvocation

    if not isinstance(invocation, ResolvedWorkflowInvocation):
        return

    console = Console()
    summary = Table(title="Workflow Resolution", show_header=True, header_style="bold cyan")
    summary.add_column("Property", style="cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("Requested", resolved_name)
    summary.add_row("Workflow", invocation.workflow_name)
    summary.add_row("Profile", invocation.profile_name or "<none>")
    summary.add_row("Force", str(invocation.force))
    summary.add_row("Steps", str(len(invocation.workflow.steps)))
    console.print(summary)

    if invocation.values:
        console.print(Panel(json.dumps(invocation.values, indent=2, default=str), title="Effective Values"))

    steps = Table(title="Workflow Steps", show_header=True, header_style="bold green")
    steps.add_column("Id", style="cyan", no_wrap=True)
    steps.add_column("Uses", style="white")
    steps.add_column("Needs", style="magenta")
    steps.add_column("Concurrency", style="yellow")
    for step in invocation.workflow.steps:
        steps.add_row(step.id, step.uses, ", ".join(step.needs) or "-", step.concurrency)
    console.print(steps)


class WorkflowCommands(CliTopCommand):
    """Workflow-related CLI commands."""

    description: str = "Workflow execution and inspection"

    def get_description(self) -> tuple[str, str]:
        return "workflow", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            workflow_or_profile: Annotated[str, typer.Argument(help="Workflow name or workflow profile name")],
            profile: Annotated[
                str | None,
                typer.Option("--profile", help="Workflow profile to use when the positional argument is a workflow"),
            ] = None,
            set_values: Annotated[
                list[str] | None,
                typer.Option("--set", help="Override values using KEY=VALUE syntax", metavar="KEY=VALUE"),
            ] = None,
            pathspec: Annotated[
                list[str] | None,
                typer.Option(
                    "--pathspec",
                    "-p",
                    help="Gitwildmatch pattern; maps to values.pathspecs (repeatable, prefix ! to exclude)",
                ),
            ] = None,
            to: Annotated[
                str | None,
                typer.Option("--to", help="Output directory; maps to values.output_dir"),
            ] = None,
            base_dir: Annotated[
                str | None,
                typer.Option("--base-dir", help="Base directory; maps to values.base_dir"),
            ] = None,
            force: Annotated[
                bool, typer.Option("--force", help="Force recomputation even if caches are valid")
            ] = False,
            dry_run: Annotated[bool, typer.Option("--dry-run", help="Resolve the workflow and print the plan")] = False,
        ) -> None:
            """Resolve and execute a workflow or workflow profile.

            Shorthand options map to workflow values:
            - ``--pathspec`` → ``values.pathspecs`` (repeatable, prefix ! to exclude)
            - ``--to`` → ``values.output_dir``
            - ``--base-dir`` → ``values.base_dir``

            These are equivalent to using ``--set KEY=VALUE`` but more convenient.

            Examples:
                ```bash
                # Run a named profile
                cli workflow run markdownize_docs --dry-run

                # Ad-hoc invocation with shorthands
                cli workflow run markdownize \\
                    --base-dir '${paths.rfq_pdf}' \\
                    --pathspec '**/*.pdf' --pathspec '!**/*_draft*' \\
                    --to '${paths.rfq_md}/real'

                # Override via --set (verbose equivalent)
                cli workflow run markdownize_docs \\
                    --set base_dir='${paths.rfq_pdf}' \\
                    --set output_dir='${paths.rfq_md}/real'
                ```
            """
            console = Console()
            try:
                cli_overrides = parse_cli_overrides(set_values)
                if pathspec:
                    cli_overrides["pathspecs"] = pathspec
                if to:
                    cli_overrides["output_dir"] = to
                if base_dir:
                    cli_overrides["base_dir"] = base_dir
                invocation = resolve_workflow_invocation(
                    workflow_or_profile,
                    profile_name=profile,
                    cli_overrides=cli_overrides,
                    force=force,
                )
            except WorkflowResolutionError as exc:
                console.print(Panel(str(exc), title="Workflow Resolution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                # Catch any other config/interpolation errors and display them nicely
                console.print(
                    Panel(
                        f"Unexpected error during workflow resolution: {type(exc).__name__}: {exc}",
                        title="Configuration Error",
                        border_style="red",
                    )
                )
                raise typer.Exit(1) from exc

            _render_workflow_summary(workflow_or_profile, invocation)

            if dry_run:
                console.print(Panel("Dry run complete — no execution performed.", border_style="green"))
                return

            from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow

            try:
                results = execute_workflow(invocation)
                console.print(Panel(f"Workflow completed: {len(results)} step(s) executed.", border_style="green"))
            except WorkflowExecutionError as exc:
                console.print(Panel(str(exc), title="Workflow Execution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                # Catch any other execution errors and display them nicely
                console.print(
                    Panel(
                        f"Unexpected error during workflow execution: {type(exc).__name__}: {exc}",
                        title="Execution Error",
                        border_style="red",
                    )
                )
                raise typer.Exit(1) from exc

        @cli_app.command("list")
        def list_items(
            kind: Annotated[
                str,
                typer.Argument(help="What to list: 'workflows', 'profiles', or 'all'"),
            ] = "all",
        ) -> None:
            """List available workflows and/or workflow profiles."""
            from genai_tk.workflow.resolver import list_workflow_names, list_workflow_profile_names

            console = Console()

            if kind in ("workflows", "all"):
                names = list_workflow_names()
                tbl = Table(title="Workflows", show_header=True, header_style="bold cyan")
                tbl.add_column("Name", style="green")
                for n in names:
                    tbl.add_row(n)
                if not names:
                    tbl.add_row("[dim]<none configured>[/dim]")
                console.print(tbl)

            if kind in ("profiles", "all"):
                names = list_workflow_profile_names()
                tbl = Table(title="Workflow Profiles", show_header=True, header_style="bold cyan")
                tbl.add_column("Name", style="green")
                for n in names:
                    tbl.add_row(n)
                if not names:
                    tbl.add_row("[dim]<none configured>[/dim]")
                console.print(tbl)
