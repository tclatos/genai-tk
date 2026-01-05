"""BAML-based version of structured extraction CLI commands.



Usage Examples:
    ```bash
    # Extract structured data from Markdown files using BAML
    uv run cli baml extract file.md --baml ReviewedOpportunity:ExtractRainbow --force

    # Process recursively with custom settings
    uv run cli baml extract ./reviews/ --recursive --batch-size 10 --force --baml ReviewedOpportunity:ExtractRainbow
    ```

Data Flow:
    1. Markdown files → BAML function → model instances
    2. Model instances → KV Store → JSON structured data
    3. Processed data → Available for EKG agent querying
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import BaseModel
from upath import UPath

from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.main.cli import CliTopCommand

LLM_ID = None
KV_STORE_ID = "default"


def check_kvstore_key_exists(key: str, model_cls: type[BaseModel], kvstore_id: str = KV_STORE_ID) -> bool:
    """Check if a key already exists in the KV store."""
    from genai_tk.utils.pydantic.kv_store import PydanticStore

    store = PydanticStore(kvstore_id=kvstore_id, model=model_cls)
    cached_obj = store.load_object(key)
    return cached_obj is not None


def save_to_kvstore(key: str, obj: BaseModel, kvstore_id: str = KV_STORE_ID) -> None:
    """Save a Pydantic object to the KV store."""
    from genai_tk.utils.pydantic.kv_store import PydanticStore

    store = PydanticStore(kvstore_id=kvstore_id, model=obj.__class__)
    store.save_obj(key, obj)
    logger.success(f"Saved to KV store with key: {key}")


class BamlCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "baml", "BAML (structured output) related commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            function_name: Annotated[
                str,
                typer.Argument(help="BAML function name to execute (e.g., ExtractResume)"),
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
            kvstore_key: Annotated[
                str | None,
                typer.Option(help="Key to store the result in the KV store (only for Pydantic outputs)"),
            ] = None,
            force: bool = typer.Option(False, "--force", help="Overwrite existing KV entry if key exists"),
        ) -> None:
            """Execute a BAML function with input text and print the result.

            Example:
            echo "John Doe, john@example.com" | uv run cli baml run ExtractResume
            uv run cli baml run ExtractResume --input "John Doe, john@example.com"
            uv run cli baml run FakeResume -i "John Smith; SW engineer" --kvstore-key fake_cv_john_smith
            """
            # Get input from stdin if not provided
            if input_text is None:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read()
                # else: input_text remains None, baml_invoke will handle no-param functions

            # Build parameters dict - baml_invoke will figure out parameter names
            # We use a generic key that baml_invoke will map to the actual first parameter
            param_dict = {"__input__": input_text} if input_text else {}

            # Execute the function using baml_invoke
            from genai_tk.extra.structured.baml_util import baml_invoke

            try:
                result = asyncio.run(baml_invoke(function_name, param_dict, config_name, llm))
            except Exception as e:
                logger.error(f"Failed to execute BAML function '{function_name}': {e}")
                return

            # Store in KV store if requested
            if kvstore_key:
                if not isinstance(result, BaseModel):
                    logger.error(
                        f"Cannot store to KV store: Result is not a Pydantic object (got {type(result).__name__})"
                    )
                    return

                # Check if key already exists unless force is set
                if not force and check_kvstore_key_exists(kvstore_key, type(result), KV_STORE_ID):
                    logger.error(f"Key '{kvstore_key}' already exists in KV store. Use --force to overwrite.")
                    return

                save_to_kvstore(kvstore_key, result, KV_STORE_ID)

            # Print result
            if isinstance(result, BaseModel):
                print(result.model_dump_json(indent=2))
            else:
                print(result)

        @cli_app.command("extract")
        def extract(
            file_or_dir: Annotated[
                Path,
                typer.Argument(
                    help="Markdown files or directories to process",
                    exists=True,
                    file_okay=True,
                    dir_okay=True,
                ),
            ],
            recursive: bool = typer.Option(False, help="Search for files recursively"),
            batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
            force: bool = typer.Option(False, "--force", help="Overwrite existing KV entries"),
            function_name: Annotated[
                str,
                typer.Option(
                    "--function",
                    help="BAML function name (e.g., ExtractRainbow, ExtractResume)",
                ),
            ] = "ExtractRainbow",
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help="Name of the structured config to use from yaml config (e.g., 'default', 'rainbow')",
                ),
            ] = "default",
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
        ) -> None:
            """Extract structured project data from Markdown files using BAML and save as JSON in a KV store.

            This command uses BAML-generated functions to extract data from markdown files.
            The return type is automatically deduced from the BAML function signature.

            Example:
            uv run cli baml extract file.md --function ExtractRainbow --force
            uv run cli baml extract ./reviews/ --recursive --function ExtractResume --llm gpt-4o
            """

            logger.info(f"Starting BAML-based project extraction with: {file_or_dir}")

            os.environ["BAML_LOG"] = "warn"

            # Collect all Markdown files
            all_files = []

            if file_or_dir.is_file() and file_or_dir.suffix.lower() in [".md", ".markdown"]:
                # Single Markdown file
                all_files.append(file_or_dir)
            elif file_or_dir.is_dir():
                # Directory - find Markdown files inside
                if recursive:
                    md_files = list(file_or_dir.rglob("*.[mM][dD]"))  # Case-insensitive match
                else:
                    md_files = list(file_or_dir.glob("*.[mM][dD]"))
                all_files.extend(md_files)
            else:
                logger.error(f"Invalid path: {file_or_dir} - must be a Markdown file or directory")
                return

            md_files = all_files  # All files are already Markdown files at this point

            if not md_files:
                logger.warning("No Markdown files found matching the provided patterns.")
                return

            logger.info(f"Found {len(md_files)} Markdown files to process")

            if force:
                logger.info("Force option enabled - will reprocess all files and overwrite existing KV entries")

            if llm:
                logger.info(f"Using LLM: {llm}")

            # Create BAML processor - model_cls will be deduced from first result
            from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor

            processor = BamlStructuredProcessor(
                function_name=function_name,
                config_name=config_name,
                llm=llm,
                kvstore_id=KV_STORE_ID,
                force=force,
            )
            if not md_files:
                logger.info("All files have already been processed. Use --force to reprocess.")
                return
            asyncio.run(processor.process_files(md_files, batch_size))

            logger.success(
                f"BAML-based project extraction complete. {len(md_files)} files processed. Results saved to KV Store"
            )

        @cli_app.command("prefect-extract")
        def prefect_extract(
            source: Annotated[
                str,
                typer.Argument(
                    help=(
                        "Markdown file or directory to process. "
                        "Can be a local path or a remote URI supported by UPath."
                    ),
                ),
            ],
            recursive: bool = typer.Option(False, help="Search for files recursively"),
            batch_size: int = typer.Option(5, help="Number of files to process concurrently in each batch"),
            force: bool = typer.Option(False, "--force", help="Reprocess files even if unchanged in manifest"),
            function_name: Annotated[
                str,
                typer.Option(
                    "--function",
                    help="BAML function name (e.g., ExtractRainbow, ExtractResume)",
                ),
            ] = "ExtractRainbow",
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help=(
                        "Name of the structured config to use from YAML config "
                        "(for example 'default' or 'rainbow')."
                    ),
                ),
            ] = "default",
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
        ) -> None:
            """Run BAML extraction as a Prefect flow and write JSON files.

            The Prefect-powered variant discovers Markdown files under ``source``,
            skips files that are unchanged according to a manifest stored in the
            target directory, and processes the remaining files in parallel
            batches using a thread-based task runner.

            Example:
            ```bash
            uv run cli baml prefect-extract ./reviews --recursive \
                --function ExtractRainbow --config default --force
            ```
            """

            os.environ["BAML_LOG"] = "warn"

            root = UPath(source)
            if not root.exists():
                logger.error(f"Source path does not exist: {root}")
                raise typer.Exit(1)

            if llm:
                logger.info(f"Using LLM: {llm}")

            logger.info(
                f"Starting Prefect-based BAML extraction for '{root}' "
                f"with function '{function_name}' and config '{config_name}'",
            )

            from genai_tk.extra.structured.baml_prefect_flow import baml_structured_extraction_flow

            try:
                run_flow_ephemeral(
                    baml_structured_extraction_flow,
                    source=str(root),
                    recursive=recursive,
                    batch_size=batch_size,
                    force=force,
                    function_name=function_name,
                    config_name=config_name,
                    llm=llm,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Prefect-based BAML extraction failed: {exc}")
                raise typer.Exit(1) from exc

            logger.success("Prefect-based BAML extraction completed successfully")
