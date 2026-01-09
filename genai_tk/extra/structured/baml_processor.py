"""BAML-based version of structured extraction.

This module  uses BAML to extract structured data from markdown files into any
Pydantic BaseModel provided at runtime.

"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Type, TypeVar

from loguru import logger
from pydantic import BaseModel
from upath import UPath

LLM_ID = None
KV_STORE_ID = "file"


T = TypeVar("T", bound=BaseModel)


class BamlStructuredProcessor(BaseModel, Generic[T]):
    """Processor that uses BAML for extracting structured data from documents.

    Args:
        model_cls: Pydantic model class to instantiate from BAML output.
                   Can be None if using function_name - will be deduced from first result.
        baml_function: Async callable BAML function that takes content string and returns model instance.
                       Can also be None if using function_name with baml_invoke parameters.
        function_name: BAML function name (alternative to baml_function)
        config_name: Config name for BAML client
        llm: LLM identifier for BAML
        kvstore_id: KV store identifier for caching
        force: Whether to bypass cache and reprocess all documents
    """

    model_cls: Type[T] | None = None
    baml_function: Callable[[str], Awaitable[Any]] | None = None
    function_name: str | None = None
    config_name: str = "default"
    llm: str | None = None
    kvstore_id: str = KV_STORE_ID
    force: bool = False
    _model_cls_deduced: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def abatch_analyze_documents(self, document_ids: list[str], markdown_contents: list[str]) -> list[T]:
        """Process multiple documents asynchronously with caching using BAML."""
        from genai_tk.utils.pydantic_utils.kv_store import PydanticStore, save_object_to_kvstore

        analyzed_docs: list[T] = []
        remaining_ids: list[str] = []
        remaining_contents: list[str] = []

        # Check cache first (unless force is enabled)
        # Only if model_cls is already known
        if self.kvstore_id and not self.force and self.model_cls is not None:
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

        # Create concurrent tasks for all remaining documents
        if self.baml_function:
            # Use provided baml_function
            tasks = [self.baml_function(content) for content in remaining_contents]
        elif self.function_name:
            # Use baml_invoke
            from genai_tk.extra.structured.baml_util import baml_invoke

            tasks = [
                baml_invoke(self.function_name, {"__input__": content}, self.config_name, self.llm)
                for content in remaining_contents
            ]
        else:
            raise ValueError("Either baml_function or function_name must be provided")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and save to KV store
        for doc_id, result in zip(remaining_ids, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f"Failed to process document {doc_id}: {result}")
                continue

            try:
                # Deduce model_cls from first successful result if not provided
                if self.model_cls is None and not self._model_cls_deduced:
                    if not isinstance(result, BaseModel):
                        raise ValueError(f"BAML function must return a Pydantic model, got {type(result).__name__}")
                    self.model_cls = type(result)
                    self._model_cls_deduced = True
                    logger.info(f"Deduced model class: {self.model_cls.__name__}")

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
