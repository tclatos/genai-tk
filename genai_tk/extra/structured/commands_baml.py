"""BAML-based version of structured extraction CLI commands.

Usage Examples:
    ```bash
    # Extract structured data from Markdown files using BAML
    uv run cli baml extract ./docs ./output --recursive --function ExtractRainbow --force

    # Use config variables for paths
    uv run cli baml extract '${paths.data_root}/reviews' '${paths.data_root}/structured' \\
        --recursive --batch-size 10 --force --function ExtractRainbow

    # Custom include/exclude patterns
    uv run cli baml extract ./reports ./output \\
        --include 'report_*.md' --include 'summary_*.md' \\
        --exclude '*_draft.md' --recursive

    # Run BAML function on single input
    uv run cli baml run FakeResume -i "John Smith; SW engineer"

    # Save result to file
    uv run cli baml run FakeResume -i "John Smith; SW engineer" \\
        --out-dir '${paths.data_root}' --out-file fake_cv_john_smith.json
    ```

Data Flow:
    1. Markdown files → BAML function → model instances
    2. Model instances → JSON structured data → output directory
    3. Manifest tracks processed files → enables incremental processing
"""

import os
import sys
from typing import Annotated

import typer
from loguru import logger
from pydantic import BaseModel

from genai_tk.cli.base import CliTopCommand


class BamlCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "baml", "BAML (structured output) related commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            function_name: Annotated[
                str,
                typer.Argument(help="BAML function name to execute (e.g., ExtractResume, FakeResume)"),
            ],
            input_text: Annotated[
                str | None,
                typer.Option("--input", "-i", help="Input text to process (if not provided, reads from stdin)"),
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
            force: bool = typer.Option(False, "--force", help="Overwrite existing output if it exists"),
        ) -> None:
            """Execute a BAML function with input text and print/save the result.

            The result is saved to a directory structure: output_dir/ModelName/output_file.
            If output_dir or output_file are not specified, the result is printed to stdout.
            A manifest.json file tracks processed inputs to avoid reprocessing.

            Examples:
                ```bash
                # Output to stdout
                uv run cli baml run FakeResume -i "John Smith; SW engineer"

                # Save to file with config variables
                uv run cli baml run FakeResume -i "John Smith; SW engineer" \\
                    --out-dir '${paths.data_root}' --out-file fake_cv_john_smith.json

                # Read from stdin and save to file
                echo "John Doe, software architect" | uv run cli baml run ExtractResume \\
                    --out-dir ./output --out-file john_doe.json

                # Force reprocessing
                uv run cli baml run FakeResume -i "John Smith; SW engineer" \\
                    --out-dir ./output --out-file fake_cv.json --force
                ```
            """
            # Validate output parameters
            if (output_dir and not output_file) or (output_file and not output_dir):
                logger.error("Both --out-dir and --out-file must be specified together, or both omitted")
                raise typer.Exit(1)

            if output_file and not output_file.endswith(".json"):
                logger.error("Output filename must have .json extension")
                raise typer.Exit(1)

            # Get input from stdin if not provided
            if input_text is None:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read().strip()
                else:
                    logger.error("No input provided. Use --input/-i or pipe input via stdin")
                    raise typer.Exit(1)

            if not input_text:
                logger.error("Input text cannot be empty")
                raise typer.Exit(1)

            if llm:
                logger.info("Using LLM: {}", llm)

            logger.info("Executing BAML function '{}' with config '{}'", function_name, config_name)

            # Execute using Prefect flow
            from genai_tk.extra.prefect.runtime import run_flow_ephemeral
            from genai_tk.extra.flows.baml_flow import baml_single_input_flow

            try:
                result, model_name = run_flow_ephemeral(
                    baml_single_input_flow,
                    input_text=input_text,
                    function_name=function_name,
                    config_name=config_name,
                    llm=llm,
                    output_dir=output_dir,
                    output_file=output_file,
                    force=force,
                )

                # Print result to stdout if no output file was specified
                if not output_dir or not output_file:
                    if isinstance(result, BaseModel):
                        print(result.model_dump_json(indent=2))
                    else:
                        import json

                        print(json.dumps(result, indent=2, default=str))
                else:
                    logger.success(
                        f"Result saved to {output_dir}/{model_name or function_name}/{output_file}",
                    )

            except Exception as exc:
                logger.error("BAML function execution failed: {}", exc)
                raise typer.Exit(1) from exc

        @cli_app.command("extract")
        def extract(
            base_dir: Annotated[
                str,
                typer.Argument(
                    help=("Root directory to walk. Supports \${paths.*} config vars."),
                ),
            ],
            output_dir: Annotated[
                str,
                typer.Argument(
                    help=(
                        "Output directory for extracted data and manifest. "
                        "Supports \${paths.*} config vars."
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
            """Extract structured data from files using BAML.

            Process files matched by pathspecs and save extracted structured data as
            JSON files to output_dir.  A manifest tracks processed files.

            Examples:
                ```bash
                cli baml extract ./docs ./output --function ExtractRainbow

                cli baml extract '\${paths.data_root}/reviews' '\${paths.data_root}/structured' \\
                    --pathspec '**/*.md' --function ExtractRainbow

                cli baml extract ./reports ./output \\
                    --pathspec '**/*.md' --pathspec '!**/*_draft.md' \\
                    --force --function ExtractResume
                ```
            """

            os.environ["BAML_LOG"] = "warn"

            if llm:
                logger.info("Using LLM: {}", llm)

            logger.info(
                "Starting BAML extraction from '{}' to '{}' with function '{}' and config '{}'",
                base_dir,
                output_dir,
                function_name,
                config_name,
            )

            from genai_tk.extra.flows.baml_flow import baml_structured_extraction_flow
            from genai_tk.utils.prefect_run import run_flow_ephemeral

            try:
                run_flow_ephemeral(
                    baml_structured_extraction_flow,
                    base_dir=base_dir,
                    output_dir=output_dir,
                    pathspecs=pathspec,
                    batch_size=batch_size,
                    force=force,
                    function_name=function_name,
                    config_name=config_name,
                    llm=llm,
                )
            except Exception as exc:
                logger.error("BAML extraction failed: {}", exc)
                raise typer.Exit(1) from exc

            logger.success("BAML extraction completed successfully")
