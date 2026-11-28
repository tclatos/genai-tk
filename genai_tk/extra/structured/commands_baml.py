"""BAML-based structured extraction CLI commands.

These commands delegate to the workflow engine for composable, cacheable execution.
See config/workflows.yaml for 'baml_extract' and 'baml_run' workflow definitions.

Usage Examples:
    ```bash
    # Extract structured data from Markdown files using BAML
    uv run cli baml extract ./docs ./output --function ExtractRainbow --force

    # Extract structured data from PDF files using BAML
    uv run cli baml extract ./scans ./output --pathspec '**/*.pdf' --function ExtractDeliveryNotePdf

    # Use config variables for paths
    uv run cli baml extract '${paths.data_root}/reviews' '${paths.data_root}/structured' \\
        --batch-size 10 --force --function ExtractRainbow

    # Run BAML function on single input
    uv run cli baml run ExtractResume -i "John Smith; SW engineer"

    # Save result to file
    uv run cli baml run ExtractResume -i "John Smith; SW engineer" \\
        --out-dir '${paths.data_root}' --out-file result.json

    # Use named workflow preset
    uv run cli baml extract with preset --set force_rebuild=true
    ```

Note:
    These commands now delegate to 'cli workflow run baml_extract' and 'cli workflow run baml_run'
    for composability, caching, and integration with the workflow engine. Direct calls to
    Prefect flows are no longer used from the CLI.
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from genai_tk.cli.base import CliTopCommand
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    resolve_workflow_invocation,
)


class BamlCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "baml", "BAML (structured output) related commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            function_name: Annotated[
                str | None,
                typer.Argument(help="BAML function name to execute (e.g., ExtractRailwayObjects, ExtractResume)"),
            ] = None,
            input_text: Annotated[
                str | None,
                typer.Option(
                    "--input", "-i", help="Input text or file path to process (if not provided, reads from stdin)"
                ),
            ] = None,
            file_path: Annotated[
                str | None,
                typer.Option("--file", "-f", help="Input file path (PDF, text, Markdown) to process"),
            ] = None,
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help="Name of the structured config to use from yaml config",
                ),
            ] = "default",
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
            output_dir: Annotated[
                str | None,
                typer.Option(
                    "--out-dir",
                    help=(
                        "Output directory for result (supports config variables like ${paths.data_root}). "
                        "If not specified, outputs to stdout."
                    ),
                ),
            ] = None,
            output_file: Annotated[
                str | None,
                typer.Option(
                    "--out-file",
                    help="Output filename (must end with .json). If not specified, outputs to stdout.",
                ),
            ] = None,
            force: bool = typer.Option(False, "--force", help="Force reprocessing even if result exists"),
        ) -> None:
            """Execute a BAML function with input text via the workflow engine.

            This command delegates to the 'baml_run' workflow for composability and caching.
            The result is saved to output_dir/ModelName/output_file or printed to stdout.
            A manifest.json file tracks processed inputs to avoid reprocessing.

            Examples:
                ```bash
                # Output to stdout
                uv run cli baml run ExtractResume -i "John Smith; SW engineer"

                # Run on a PDF file
                uv run cli baml run ExtractRailwayObjects -i ./diagram.pdf

                # Save to file with config variables
                uv run cli baml run ExtractResume -i "John Smith; SW engineer" \\
                    --out-dir '${paths.data_root}' --out-file result.json

                # Read from stdin and save to file
                echo "John Doe, software architect" | uv run cli baml run ExtractResume \\
                    --out-dir ./output --out-file john_doe.json

                # Force reprocessing
                uv run cli baml run ExtractResume -i "John Smith; SW engineer" \\
                    --out-dir ./output --out-file result.json --force
                ```
            """
            console = Console()

            if file_path:
                input_text = file_path

            if output_dir and not output_file:
                # Default output filename from input file stem or 'result.json'
                cand_name = file_path or input_text or function_name
                if cand_name and Path(os.path.expandvars(cand_name)).is_file():
                    output_file = f"{Path(os.path.expandvars(cand_name)).stem}.json"
                else:
                    output_file = f"{function_name or 'result'}.json"
            elif output_file and not output_dir:
                output_dir = "."

            if output_file and not output_file.endswith(".json"):
                console.print("[red]Error: Output filename must have .json extension[/red]")
                raise typer.Exit(1)

            # If function_name is passed as a file path and input_text was not specified, treat it as input_text
            if function_name:
                cand = Path(os.path.expandvars(function_name)).expanduser()
                if cand.is_file() or any(
                    function_name.lower().endswith(ext) for ext in [".pdf", ".md", ".txt", ".json"]
                ):
                    if input_text is None:
                        input_text = function_name
                        function_name = None

            # Get input from stdin if piped
            if input_text is None and not sys.stdin.isatty():
                input_text = sys.stdin.read().strip()

            # Try to resolve defaults from structured config if not provided
            if function_name is None:
                try:
                    from genai_tk.config_mgmt.config_mngr import global_config
                    from genai_tk.extra.structured.baml_util import StructuredConfig

                    configs = global_config().section_dict("structured", StructuredConfig, inject_name=False)
                    if config_name in configs and getattr(configs[config_name], "default_function", None):
                        function_name = configs[config_name].default_function
                except Exception:
                    pass

            if llm is None:
                try:
                    from genai_tk.config_mgmt.config_mngr import global_config
                    from genai_tk.extra.structured.baml_util import StructuredConfig

                    configs = global_config().section_dict("structured", StructuredConfig, inject_name=False)
                    if config_name in configs and getattr(configs[config_name], "llm", None):
                        llm = configs[config_name].llm
                except Exception:
                    pass
                if llm is None:
                    llm = "gemini-3.8-flash@openrouter"

            # Build workflow invocation parameters
            cli_overrides: dict[str, Any] = {
                "config_name": config_name,
                "force": force,
            }
            if input_text is not None:
                cli_overrides["input_text"] = input_text
            if function_name is not None:
                cli_overrides["function_name"] = function_name
            if llm is not None:
                cli_overrides["llm"] = llm
            if output_dir is not None:
                cli_overrides["output_dir"] = output_dir
            if output_file is not None:
                cli_overrides["output_file"] = output_file

            # Resolve and run the workflow
            try:
                invocation = resolve_workflow_invocation("baml_run", cli_overrides=cli_overrides, force=force)
            except WorkflowResolutionError as exc:
                console.print(Panel(str(exc), title="Workflow Resolution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(
                    Panel(
                        f"Unexpected error during workflow resolution: {type(exc).__name__}: {exc}",
                        title="Error",
                        border_style="red",
                    ),
                )
                raise typer.Exit(1) from exc

            # Run the workflow
            try:
                from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow

                results = execute_workflow(invocation)
                step_result = next(iter(results.values()), None)
                resolved_llm = None
                output_model_name = None

                if isinstance(step_result, tuple):
                    if len(step_result) >= 2:
                        output_model_name = step_result[1]
                    if len(step_result) >= 3:
                        resolved_llm = step_result[2]
                    relative_output_path = step_result[3] if len(step_result) >= 4 else None
                else:
                    relative_output_path = None

                # Print result to stdout if no output file was specified
                if not output_dir or not output_file:
                    # results is a dict of step_id -> return value; baml_run has one step "run"
                    # baml_single_input_flow returns (result, model_name, resolved_llm)
                    if isinstance(step_result, tuple):
                        step_result = step_result[0]
                    if step_result is not None:
                        if hasattr(step_result, "model_dump_json"):
                            print(step_result.model_dump_json(indent=2))
                        else:
                            print(json.dumps(step_result, indent=2, default=str))
                else:
                    console.print(
                        f"[green]BAML run complete.[/green]\n"
                        f"  LLM     : {resolved_llm or 'default'}\n"
                        f"  Schema  : {output_model_name or 'n/a'}\n"
                        f"  Output  : {output_dir}/{relative_output_path or output_file}"
                    )

            except WorkflowExecutionError as exc:
                console.print(Panel(str(exc), title="Workflow Execution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(
                    Panel(
                        f"Unexpected error during workflow execution: {type(exc).__name__}: {exc}",
                        title="Execution Error",
                        border_style="red",
                    ),
                )
                raise typer.Exit(1) from exc

        @cli_app.command("extract")
        def extract(
            base_dir: Annotated[
                str,
                typer.Argument(
                    help=(r"Root directory to walk. Supports \${paths.*} config vars."),
                ),
            ],
            output_dir: Annotated[
                str,
                typer.Argument(
                    help=(
                        "Output directory for extracted data and manifest. "
                        r"Supports \${paths.*} config vars."
                    ),
                ),
            ],
            function_name: Annotated[
                str,
                typer.Option(
                    "--function",
                    help="BAML function name (e.g., ExtractRainbow, ExtractResume)",
                ),
            ],
            pathspec: Annotated[
                list[str] | None,
                typer.Option(
                    "--pathspec",
                    "-p",
                    help="Gitwildmatch pattern (repeatable; prefix ! to exclude). Default: **/*.md",
                ),
            ] = None,
            batch_size: int = typer.Option(5, help="Number of files to process concurrently in each batch"),
            force: bool = typer.Option(False, "--force", help="Reprocess files even if unchanged in manifest"),
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help="Name of the structured config to use from YAML config (e.g., 'default', 'rainbow')",
                ),
            ] = "default",
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
        ) -> None:
            """Extract structured data from files using BAML via the workflow engine.

            This command delegates to the 'baml_extract' workflow for composability and caching.
            Process files matched by pathspecs and save extracted structured data as
            JSON files to output_dir.  A manifest tracks processed files.

            Supported file types:
            - Markdown (.md, .markdown): passed as plain text to the BAML function.
            - PDF (.pdf): passed as a native pdf media object to the BAML function.

            Examples:
                ```bash
                cli baml extract ./docs ./output --function ExtractRainbow

                cli baml extract '${paths.data_root}/reviews' '${paths.data_root}/structured' \\
                    --pathspec '**/*.md' --function ExtractRainbow

                cli baml extract ./reports ./output \\
                    --pathspec '**/*.md' --pathspec '!**/*_draft.md' \\
                    --force --function ExtractResume
                ```
            """
            console = Console()

            logger.info(
                "Starting BAML extraction workflow from '{}' to '{}' with function '{}' and config '{}'",
                base_dir,
                output_dir,
                function_name,
                config_name,
            )

            # Build workflow invocation parameters
            cli_overrides = {
                "base_dir": base_dir,
                "output_dir": output_dir,
                "function_name": function_name,
                "config_name": config_name,
                "batch_size": batch_size,
                "force": force,
            }

            if pathspec:
                cli_overrides["pathspecs"] = pathspec

            if llm:
                cli_overrides["llm"] = llm

            # Resolve and run the workflow
            try:
                invocation = resolve_workflow_invocation("baml_extract", cli_overrides=cli_overrides, force=force)
            except WorkflowResolutionError as exc:
                console.print(Panel(str(exc), title="Workflow Resolution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(
                    Panel(
                        f"Unexpected error during workflow resolution: {type(exc).__name__}: {exc}",
                        title="Error",
                        border_style="red",
                    ),
                )
                raise typer.Exit(1) from exc

            # Run the workflow
            try:
                from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow

                results = execute_workflow(invocation)
                step_result = next(iter(results.values()), None)

                if hasattr(step_result, "resolved_llm"):
                    resolved_llm = step_result.resolved_llm
                    output_model_name = getattr(step_result, "model_name", None)
                    entry_count = len(getattr(step_result, "entries", {}) or {})
                    console.print(
                        f"[green]BAML extraction completed.[/green]\n"
                        f"  LLM     : {resolved_llm or 'default'}\n"
                        f"  Schema  : {output_model_name or 'n/a'}\n"
                        f"  Files   : {entry_count}\n"
                        f"  Output  : {output_dir}"
                    )
                else:
                    logger.success("BAML extraction completed successfully")

            except WorkflowExecutionError as exc:
                console.print(Panel(str(exc), title="Workflow Execution Error", border_style="red"))
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(
                    Panel(
                        f"Unexpected error during workflow execution: {type(exc).__name__}: {exc}",
                        title="Execution Error",
                        border_style="red",
                    ),
                )
                raise typer.Exit(1) from exc
