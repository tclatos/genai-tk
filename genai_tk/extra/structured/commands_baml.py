"""BAML-based version of structured extraction CLI commands.

This module provides an alternative implementation of the structured_extract function
that uses BAML instead of langchain for structured output extraction. It leverages
the BAML-generated client to extract structured data from markdown files into any
Pydantic BaseModel provided at runtime.

Key Features:
    - Uses any BAML-generated function for structured data extraction
    - Maintains the same CLI interface as the original version
    - Compatible with existing KV store and batch processing infrastructure
    - Supports any Pydantic BaseModel (validated at runtime)

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
import importlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Generic, Type, TypeVar

import typer
from loguru import logger
from pydantic import BaseModel
from upath import UPath

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config

LLM_ID = None
KV_STORE_ID = "file"


T = TypeVar("T", bound=BaseModel)


class BamlStructuredProcessor(BaseModel, Generic[T]):
    """Processor that uses BAML for extracting structured data from documents.

    Args:
        model_cls: Pydantic model class to instantiate from BAML output
        baml_function: Async callable BAML function that takes content string and returns model instance
        kvstore_id: KV store identifier for caching
        force: Whether to bypass cache and reprocess all documents
    """

    model_cls: Type[T]
    baml_function: Callable[[str], Awaitable[Any]]
    kvstore_id: str = KV_STORE_ID
    force: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def abatch_analyze_documents(self, document_ids: list[str], markdown_contents: list[str]) -> list[T]:
        """Process multiple documents asynchronously with caching using BAML."""
        from genai_tk.utils.pydantic.kv_store import PydanticStore, save_object_to_kvstore

        analyzed_docs: list[T] = []
        remaining_ids: list[str] = []
        remaining_contents: list[str] = []

        # Check cache first (unless force is enabled)
        if self.kvstore_id and not self.force:
            for doc_id, content in zip(document_ids, markdown_contents, strict=True):
                cached_doc = PydanticStore(kvstore_id=self.kvstore_id, model=self.model_cls).load_object(doc_id)

                if cached_doc:
                    analyzed_docs.append(cached_doc)
                    logger.info(f"Loaded cached document: {doc_id}")
                else:
                    remaining_ids.append(doc_id)
                    remaining_contents.append(content)
        else:
            remaining_ids = document_ids
            remaining_contents = markdown_contents

        if not remaining_ids:
            return analyzed_docs

        # Process uncached documents using BAML concurrent calls pattern
        logger.info(f"Processing {len(remaining_ids)} documents with BAML async client...")

        # Create concurrent tasks for all remaining documents using the provided BAML function
        tasks = [self.baml_function(content) for content in remaining_contents]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and save to KV store
        for doc_id, result in zip(remaining_ids, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f"Failed to process document {doc_id}: {result}")
                continue

            try:
                # Add document_id as a custom attribute
                result_dict = result.model_dump()
                result_dict["document_id"] = doc_id
                result_with_id = self.model_cls(**result_dict)

                analyzed_docs.append(result_with_id)
                logger.success(f"Processed document: {doc_id}")

                # Save to KV store
                if self.kvstore_id:
                    save_object_to_kvstore(doc_id, result_with_id, kv_store_id=self.kvstore_id)
                    logger.debug(f"Saved to KV store: {doc_id}")

            except Exception as e:
                logger.error(f"Failed to save document {doc_id}: {e}")

        return analyzed_docs

    def analyze_document(self, document_id: str, markdown: str) -> T:
        """Analyze a single document synchronously using BAML."""
        try:
            results = asyncio.run(self.abatch_analyze_documents([document_id], [markdown]))
        except RuntimeError:
            # If we're in an async context, try nest_asyncio
            try:
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_running_loop()
                results = loop.run_until_complete(self.abatch_analyze_documents([document_id], [markdown]))
            except Exception as e:
                raise ValueError(f"Failed to process document {document_id}: {e}") from e

        if results:
            return results[0]
        else:
            raise ValueError(f"Failed to process document: {document_id}")

    async def process_files(self, md_files: list[UPath], batch_size: int = 5) -> None:
        """Process markdown files in batches using BAML."""
        document_ids = []
        markdown_contents = []
        valid_files = []

        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                document_ids.append(file_path.stem)
                markdown_contents.append(content)
                valid_files.append(file_path)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not document_ids:
            logger.warning("No valid files to process")
            return

        logger.info(f"Processing {len(valid_files)} files using BAML. Output in '{self.kvstore_id}' KV Store")

        # Process all documents (BAML handles batching internally)
        _ = await self.abatch_analyze_documents(document_ids, markdown_contents)


def _load_baml_client(config_name: str = "default") -> tuple[Any, Any]:
    """Load BAML client modules dynamically from config.

    Args:
        config_name: Name of the structured config to use (e.g., 'default', 'rainbow')

    Returns:
        Tuple of (types_module, async_client_instance)
    """
    # Get BAML client package path from config
    config_key = f"structured.{config_name}.baml_client"
    baml_client_package = global_config().get_str(config_key)

    if not baml_client_package:
        raise ValueError(
            f"BAML client package not found in config at '{config_key}'. "
            f"Please configure it in YAML config file (overrides.yaml or else)"
        )

    logger.debug(f"Loading BAML client from package: {baml_client_package}")

    # Dynamically import the types module
    try:
        types_module = importlib.import_module(f"{baml_client_package}.types")
    except ImportError as e:
        raise ImportError(f"Failed to import types module from '{baml_client_package}.types': {e}") from e

    # Dynamically import the async client
    try:
        async_client_module = importlib.import_module(f"{baml_client_package}.async_client")
        baml_async_client = async_client_module.b
    except ImportError as e:
        raise ImportError(f"Failed to import async client from '{baml_client_package}.async_client': {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Async client module does not have expected 'b' attribute: {e}") from e

    return types_module, baml_async_client


class BamlCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "baml", "BAML (structured output) related commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
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
            baml_spec: Annotated[
                str,
                typer.Option(
                    "--baml",
                    help="BAML specification in format 'ClassName:FunctionName' (e.g., ReviewedOpportunity:ExtractRainbow)",
                ),
            ] = "ReviewedOpportunity:ExtractRainbow",
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help="Name of the structured config to use from overrides.yaml (e.g., 'default', 'rainbow')",
                ),
            ] = "default",
        ) -> None:
            """Extract structured project data from Markdown files using BAML and save as JSON in a key-value store.

            This command uses BAML-generated functions to extract data from markdown files
            and instantiate the specified Pydantic model class.

            Example:
            uv run cli baml extract file.md --baml ReviewedOpportunity:ExtractRainbow --force
            uv run cli baml extract ./reviews/ --recursive --baml ReviewedOpportunity:ExtractRainbow
            """

            logger.info(f"Starting BAML-based project extraction with: {file_or_dir}")

            # Load BAML client modules dynamically from config
            try:
                baml_types, baml_async_client = _load_baml_client(config_name)
                logger.debug(f"Successfully loaded BAML client for config: {config_name}")
            except Exception as e:
                logger.error(f"Failed to load BAML client: {e}")
                return

            # Parse BAML specification (format: ClassName:FunctionName)
            try:
                class_name, function_name = baml_spec.split(":", 1)
            except ValueError:
                logger.error(
                    f"Invalid BAML specification '{baml_spec}'. Expected format: 'ClassName:FunctionName' "
                    f"(e.g., 'ReviewedOpportunity:ExtractRainbow')"
                )
                return

            # Resolve model class from the BAML types module
            try:
                model_cls = getattr(baml_types, class_name)
            except AttributeError as e:
                logger.error(f"Unknown class '{class_name}' in baml_client.types: {e}")
                return

            if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
                logger.error(f"Provided class '{class_name}' is not a Pydantic BaseModel")
                return

            # Resolve BAML function from the async client
            try:
                baml_function_method = getattr(baml_async_client, function_name)
            except AttributeError as e:
                logger.error(f"Unknown BAML function '{function_name}' in async client: {e}")
                return

            # Create a wrapper function that matches the expected signature
            async def baml_function_wrapper(content: str) -> Any:
                # BAML functions typically take the content as the first positional argument
                # The parameter name varies (e.g., 'rainbow_file' for ExtractRainbow)
                return await baml_function_method(content)

            baml_function = baml_function_wrapper

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
