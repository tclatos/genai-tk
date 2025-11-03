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
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import BaseModel

from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor
from genai_tk.extra.structured.baml_util import create_baml_client_registry, get_baml_function, load_baml_client
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.pydantic.common import validate_pydantic_model

LLM_ID = None
KV_STORE_ID = "file"


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
                    help="Name of the structured config to use from yaml confif",
                ),
            ] = "default",
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
        ) -> None:
            """Execute a BAML function with input text and print the result.

            Example:
            echo "John Doe, john@example.com" | uv run cli baml run ExtractResume
            uv run cli baml run ExtractResume --input "John Doe, john@example.com"
            """
            try:
                baml_types, baml_async_client = load_baml_client(config_name)
                logger.debug(f"Successfully loaded BAML client for config: {config_name}")
            except Exception as e:
                logger.error(f"Failed to load BAML client: {e}")
                return

            # Get BAML function and return type
            try:
                baml_function, return_type = get_baml_function(baml_async_client, function_name)
            except AttributeError as e:
                logger.error(str(e))
                return

            # Check if function requires input by inspecting its signature
            import inspect

            sig = inspect.signature(getattr(baml_async_client, function_name))
            params = [p for p in sig.parameters.keys() if p != "baml_options"]

            # Get input from stdin if not provided and function requires input
            if input_text is None and params:
                if sys.stdin.isatty():
                    logger.error("No input provided. Use --input or pipe data via stdin.")
                    return
                input_text = sys.stdin.read()

            baml_options = None
            if llm:
                baml_options = {"client_registry": create_baml_client_registry(llm)}

            # Execute the function based on its signature
            try:
                if not params:
                    # Function takes no arguments
                    result = asyncio.run(baml_function(baml_options=baml_options))
                else:
                    # Function takes at least one argument, pass input_text as the first one
                    result = asyncio.run(baml_function(input_text, baml_options=baml_options))
            except Exception as e:
                logger.error(f"Failed to execute BAML function '{function_name}': {e}")
                return

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
        ) -> None:
            """Extract structured project data from Markdown files using BAML and save as JSON in a key-value store.

            This command uses BAML-generated functions to extract data from markdown files.
            The return type is automatically deduced from the BAML function signature.

            Example:
            uv run cli baml extract file.md --function ExtractRainbow --force
            uv run cli baml extract ./reviews/ --recursive --function ExtractResume
            """

            logger.info(f"Starting BAML-based project extraction with: {file_or_dir}")

            # Load BAML client modules dynamically from config
            try:
                baml_types, baml_async_client = load_baml_client(config_name)
                logger.debug(f"Successfully loaded BAML client for config: {config_name}")
            except Exception as e:
                logger.error(f"Failed to load BAML client: {e}")
                return

            # Get BAML function and deduce return type
            try:
                baml_function, model_cls = get_baml_function(baml_async_client, function_name)
            except AttributeError as e:
                logger.error(str(e))
                return

            # Validate that the return type is a Pydantic model
            if model_cls is None:
                logger.error(f"Could not deduce return type from BAML function '{function_name}'")
                return

            try:
                model_cls = validate_pydantic_model(model_cls, function_name)
            except ValueError as e:
                logger.error(f"BAML function '{function_name}' must return a Pydantic BaseModel: {e}")
                return

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

            # Create BAML processor
            processor = BamlStructuredProcessor(
                model_cls=model_cls, baml_function=baml_function, kvstore_id=KV_STORE_ID, force=force
            )

            # Filter out files that already have JSON in KV unless forced
            if not force:
                from genai_tk.utils.pydantic.kv_store import PydanticStore

                unprocessed_files = []
                for md_file in md_files:
                    key = md_file.stem
                    cached_doc = PydanticStore(kvstore_id=KV_STORE_ID, model=model_cls).load_object(key)
                    if not cached_doc:
                        unprocessed_files.append(md_file)
                    else:
                        logger.info(f"Skipping {md_file.name} - JSON already exists (use --force to overwrite)")
                md_files = unprocessed_files

            if not md_files:
                logger.info("All files have already been processed. Use --force to reprocess.")
                return

            asyncio.run(processor.process_files(md_files, batch_size))

            logger.success(
                f"BAML-based project extraction complete. {len(md_files)} files processed. Results saved to KV Store"
            )
