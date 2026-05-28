"""CLI commands for the workflow configuration layer."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genai_tk.cli.base import CliTopCommand
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    load_workflows,
    parse_cli_overrides,
    resolve_workflow_invocation,
)


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
    summary.add_row("Preset", invocation.profile_name or "<none>")
    summary.add_row("Force", str(invocation.force))
    summary.add_row("Steps", str(len(invocation.workflow.steps)))
    console.print(summary)

    if invocation.values:
        console.print(Panel(json.dumps(invocation.values, indent=2, default=str), title="Effective Values"))

    steps = Table(title="Workflow Steps", show_header=True, header_style="bold green")
    steps.add_column("Id", style="cyan", no_wrap=True)
    steps.add_column("Invoke", style="white")
    steps.add_column("Wait For", style="magenta")
    for step in invocation.workflow.steps:
        target = step.invoke.target if step.invoke else "-"
        wait_for = ", ".join(step.wait_for) if step.wait_for else "-"
        steps.add_row(step.id, target, wait_for)
    console.print(steps)


def _render_cache_status(invocation: object, console: Console) -> None:
    """Probe the workflow manifest and show per-step cache freshness."""
    from genai_tk.workflow.models import ResolvedWorkflowInvocation

    if not isinstance(invocation, ResolvedWorkflowInvocation):
        return

    from genai_tk.workflow.compiler import WorkflowCompiler
    from genai_tk.workflow.flow_cache.manifest import ManifestCache
    from genai_tk.workflow.prefect.flow_factory import _prepare_inputs, compute_step_fingerprint, workflow_manifest_path

    try:
        compiled = WorkflowCompiler().compile(invocation.workflow, invocation.values)
    except Exception as exc:
        console.print(f"[yellow]Cache status unavailable (compilation error): {exc}[/yellow]")
        return

    manifest_path = workflow_manifest_path(compiled.name)
    manifest = ManifestCache.load(manifest_path)
    force = bool(invocation.values.get("force") or invocation.values.get("force_rebuild"))

    table = Table(title="Cache Status", show_header=True, header_style="bold blue")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Backend", style="dim")
    table.add_column("Status", no_wrap=True)
    table.add_column("Fingerprint", style="dim", no_wrap=True)
    table.add_column("Last Run", style="dim")

    for step in compiled.steps:
        # In dry-run we can't resolve ${steps.*} refs (no prior results), so
        # we pass an empty results dict — those refs resolve to None and are
        # stripped, giving a best-effort fingerprint.
        step_inputs = _prepare_inputs(step.with_, {})
        fp = compute_step_fingerprint(step.id, step_inputs)
        backend = step.cache.backend

        if backend == "none":
            status = "[dim]no cache[/dim]"
            last_run = "-"
            fp_display = "-"
        elif force:
            status = "[yellow]FORCED[/yellow]"
            record = manifest.records.get(step.id)
            last_run = record.processed_at.strftime("%Y-%m-%d %H:%M") if record else "never"
            fp_display = fp[:8]
        elif manifest.is_fresh(step.id, fingerprint=fp):
            record = manifest.records.get(step.id)
            status = "[green]FRESH[/green]"
            last_run = record.processed_at.strftime("%Y-%m-%d %H:%M") if record else "-"
            fp_display = fp[:8]
        else:
            record = manifest.records.get(step.id)
            status = "[red]STALE[/red]" if record else "[yellow]UNKNOWN[/yellow]"
            last_run = record.processed_at.strftime("%Y-%m-%d %H:%M") if record else "never"
            fp_display = fp[:8]

        table.add_row(step.id, backend, status, fp_display, last_run)

    console.print(table)
    if manifest_path.exists():
        console.print(f"[dim]Manifest: {manifest_path}[/dim]")


class WorkflowCommands(CliTopCommand):
    """Workflow-related CLI commands."""

    description: str = "Workflow execution and inspection"

    def get_description(self) -> tuple[str, str]:
        return "workflow", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            workflow_name: Annotated[
                str,
                typer.Argument(help="Workflow name, or 'workflow_name/preset_name' to select a named preset."),
            ],
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
            """Run a workflow, optionally selecting a named preset.

            Use ``workflow_name/preset_name`` to select a preset defined inside
            the workflow.  Presets supply concrete values for the workflow's
            parameters (e.g. directory paths).

            Shorthand options map to workflow values:
            - ``--pathspec`` → ``values.pathspecs`` (repeatable, prefix ! to exclude)
            - ``--to`` → ``values.output_dir``
            - ``--base-dir`` → ``values.base_dir``

            Examples::

                # Run with a named preset
                cli workflow run markdownize/rainbow --dry-run

                # Run with ad-hoc overrides
                cli workflow run markdownize \\
                    --base-dir /data/pdfs --to /data/md --pathspec '**/*.pdf'

                # Override a value inline
                cli workflow run full_rainbow/default --set force_rebuild=true
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
                    workflow_name,
                    cli_overrides=cli_overrides,
                    force=force,
                )
            except WorkflowResolutionError as exc:
                console.print(Panel(str(exc), title="Workflow Resolution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(
                    Panel(
                        f"Unexpected error during workflow resolution: {type(exc).__name__}: {exc}",
                        title="Configuration Error",
                        border_style="red",
                    )
                )
                raise typer.Exit(1) from exc

            if dry_run:
                _render_workflow_summary(workflow_name, invocation)
                _render_cache_status(invocation, console)
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
                console.print(
                    Panel(
                        f"Unexpected error during workflow execution: {type(exc).__name__}: {exc}",
                        title="Execution Error",
                        border_style="red",
                    )
                )
                raise typer.Exit(1) from exc

        @cli_app.command("list")
        def list_items() -> None:
            """List all available workflows with their presets."""
            console = Console()

            workflows = load_workflows()
            if not workflows:
                console.print("[dim]No v2 workflows configured.[/dim]")
                return

            tbl = Table(title="Workflows", show_header=True, header_style="bold cyan", show_lines=True)
            tbl.add_column("Workflow", style="bold green", no_wrap=True)
            tbl.add_column("Presets", style="yellow")
            tbl.add_column("Steps", style="dim", justify="right")
            tbl.add_column("Description", style="white")

            for name, wf in sorted(workflows.items()):
                presets = ", ".join(sorted(wf.presets)) if wf.presets else "[dim]-[/dim]"
                step_count = "1" if wf.run else str(len(wf.pipeline))
                tbl.add_row(name, presets, step_count, wf.description or "[dim]-[/dim]")

            console.print(tbl)
            console.print("[dim]Run a workflow: cli workflow run <name>[/dim]")
            console.print("[dim]Run a preset:   cli workflow run <name>/<preset>[/dim]")

        @cli_app.command("show")
        def show(
            workflow_name: Annotated[str, typer.Argument(help="Workflow name to inspect")],
        ) -> None:
            """Show the DAG, parameters, and presets of a workflow."""
            console = Console()

            workflows = load_workflows()
            if workflow_name not in workflows:
                available = ", ".join(sorted(workflows)) or "<none>"
                console.print(
                    Panel(
                        f"Workflow '{workflow_name}' not found.\nAvailable: {available}",
                        title="Not Found",
                        border_style="red",
                    )
                )
                raise typer.Exit(1)

            wf = workflows[workflow_name]

            console.print(Panel(f"[bold]{wf.name}[/bold]  {wf.description}", title="Workflow"))

            # Steps / pipeline
            if wf.run:
                step_tbl = Table(title="Steps (single-step)", header_style="bold green", show_header=True)
                step_tbl.add_column("ID", style="cyan")
                step_tbl.add_column("Run", style="white")
                step_tbl.add_column("Cache", style="dim")
                step_tbl.add_row("run", wf.run, wf.resolved_cache().backend)
                console.print(step_tbl)
            else:
                step_tbl = Table(
                    title=f"Pipeline ({len(wf.pipeline)} step(s))", header_style="bold green", show_header=True
                )
                step_tbl.add_column("ID", style="cyan", no_wrap=True)
                step_tbl.add_column("Run", style="white")
                step_tbl.add_column("After", style="magenta")
                step_tbl.add_column("Cache", style="dim")
                for ps in wf.pipeline:
                    after = ", ".join(ps.dependencies) if ps.dependencies else "-"
                    step_tbl.add_row(ps.id, ps.run, after, ps.cache.backend)
                console.print(step_tbl)

            # Defaults
            if wf.defaults:
                console.print(Panel(json.dumps(wf.defaults, indent=2, default=str), title="Defaults"))

            # Presets
            if wf.presets:
                preset_tbl = Table(title="Presets", header_style="bold yellow", show_header=True)
                preset_tbl.add_column("Name", style="yellow", no_wrap=True)
                preset_tbl.add_column("Values", style="white")
                for pname, pvals in sorted(wf.presets.items()):
                    preset_tbl.add_row(pname, json.dumps(pvals, indent=2, default=str))
                console.print(preset_tbl)
            else:
                console.print("[dim]No presets defined. Run with: cli workflow run {name}[/dim]".format(name=wf.name))

        @cli_app.command("validate")
        def validate() -> None:
            """Validate all v2 workflow definitions: check targets resolve and DAGs are acyclic."""
            console = Console()
            from genai_tk.workflow.compiler import WorkflowCompiler
            from genai_tk.workflow.resolver import WorkflowResolutionError as _ResErr

            workflows = load_workflows()
            if not workflows:
                console.print("[yellow]No v2 workflows found.[/yellow]")
                return

            errors: list[str] = []
            ok: list[str] = []
            parameterized: list[str] = []

            for name, wf in sorted(workflows.items()):
                # Determine which invocations to validate:
                # always try the base workflow (no preset) — if required params
                # are missing without a default, skip base and validate presets.
                candidates: list[str] = [name] + [f"{name}/{p}" for p in sorted(wf.presets)]
                validated: list[str] = []
                wf_errors: list[str] = []
                base_missing_params: list[str] = []
                for candidate in candidates:
                    try:
                        spec = resolve_workflow_invocation(candidate, cli_overrides={})
                        WorkflowCompiler().compile(spec.workflow, spec.values)
                        validated.append(candidate)
                    except _ResErr as exc:
                        msg = str(exc)
                        # "missing required param" on the base workflow → record
                        # the missing params but don't count it as an error.
                        if "missing required parameter" in msg and candidate == name:
                            import re

                            m = re.search(r"missing required parameter\(s\): (.+?)\.", msg)
                            base_missing_params = m.group(1).split(", ") if m else ["?"]
                            continue
                        wf_errors.append(f"  {candidate}: {msg.splitlines()[0]}")
                    except Exception as exc:
                        wf_errors.append(f"  {candidate}: {exc}")

                if wf_errors:
                    errors.extend([f"[bold red]{name}[/bold red]"] + wf_errors)
                elif validated:
                    ok.append(f"{name} ({len(validated)} variant(s))")
                elif base_missing_params:
                    # Workflow requires params and has no directly-runnable variant
                    # (e.g. a library sub-workflow).  It is syntactically valid.
                    parameterized.append(f"{name} (requires: {', '.join(base_missing_params)})")

            if ok:
                console.print(Panel("\n".join(f"[green]✓[/green] {n}" for n in ok), title="Valid"))
            if parameterized:
                console.print(
                    Panel(
                        "\n".join(f"[yellow]~[/yellow] {n}" for n in parameterized),
                        title="Valid (parameterized — always provide values)",
                    )
                )
            if errors:
                console.print(Panel("\n".join(errors), title="Errors", border_style="red"))
                raise typer.Exit(1)
