"""Prefect-powered structured extraction for BAML outputs.

This module defines a Prefect flow that runs BAML-based structured
extraction on Markdown files, writes results as JSON files to a directory
rooted in the application configuration, and maintains a manifest file to
avoid duplicate processing.

Typical usage from the CLI is similar to:

```bash
uv run cli baml prefect /path/to/docs --function ExtractRainbow --force
```
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field
from upath import UPath

from genai_tk.extra.structured.baml_util import baml_invoke, prompt_fingerprint
from genai_tk.utils.file_patterns import resolve_files
from genai_tk.utils.hashing import buffer_digest


class BamlExtractionManifestEntry(BaseModel):
    """Single processed file entry in the extraction manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime


class BamlExtractionManifest(BaseModel):
    """Manifest of processed files for a given BAML function."""

    function_name: str
    config_name: str
    llm: str | None = None
    model_name: str | None = None
    schema_fingerprint: str | None = None
    entries: dict[str, BamlExtractionManifestEntry] = Field(default_factory=dict)


@dataclass(slots=True)
class _FileToProcess:
    path: UPath
    content_hash: str
    content_text: str


def _compute_hash(content: bytes) -> str:
    return buffer_digest(content)


def _load_manifest(manifest_path: UPath) -> BamlExtractionManifest | None:
    if not manifest_path.exists():
        return None

    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = json.loads(text)
        return BamlExtractionManifest.model_validate(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load manifest from {manifest_path}: {exc}. Ignoring it.")
        return None


def _save_manifest(manifest: BamlExtractionManifest, manifest_path: UPath) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _find_existing_manifest(
    structured_root: UPath,
    function_name: str,
    config_name: str,
    llm: str | None,
) -> tuple[BamlExtractionManifest | None, UPath | None]:
    """Locate an existing manifest for the given BAML configuration.

    Manifests are stored under ``structured/<model_name>/manifest.json``.
    This helper scans subdirectories under ``structured_root`` and returns
    the first manifest whose configuration matches.
    """

    if not structured_root.exists():
        return None, None

    for child in structured_root.iterdir():
        if not child.is_dir():
            continue
        candidate = child / "manifest.json"
        if not candidate.exists():
            continue
        manifest = _load_manifest(candidate)
        if manifest is None:
            continue
        if manifest.function_name == function_name and manifest.config_name == config_name and manifest.llm == llm:
            return manifest, candidate

    return None, None


def _iter_markdown_files(files: list[UPath]) -> Iterable[UPath]:
    """Yield markdown files from a pre-resolved list."""
    for path in files:
        if path.suffix.lower() in {".md", ".markdown"}:
            yield path


def _prepare_files(
    files: Iterable[UPath],
    manifest: BamlExtractionManifest,
    force: bool,
) -> tuple[list[_FileToProcess], int]:
    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = _compute_hash(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {path}: {exc}")
            continue

        key = str(path)
        existing = manifest.entries.get(key)

        if existing and not force and existing.source_hash == content_hash:
            skipped += 1
            logger.info(f"Skipping unchanged file: {path}")
            continue

        to_process.append(
            _FileToProcess(
                path=path,
                content_hash=content_hash,
                content_text=content_bytes.decode("utf-8", errors="replace"),
            )
        )

    return to_process, skipped


@task
async def _process_single_file_task(
    file_info: _FileToProcess,
    function_name: str,
    config_name: str,
    llm: str | None,
    structured_root: str,
    root_dir: str,
) -> tuple[str, BamlExtractionManifestEntry, str | None]:
    """Run BAML on a single file and persist result as JSON.

    Returns a tuple of (source_path, manifest_entry, model_name).
    """

    upath = file_info.path
    logger.info(f"Processing file with BAML: {upath}")

    params: dict[str, Any] = {"__input__": file_info.content_text}
    result = await baml_invoke(function_name, params, config_name, llm)

    # Determine model name and output directory. For Pydantic models we
    # use ``structured/<model_name>/`` as the root; otherwise we fall
    # back to ``structured/<function_name>/``.
    if isinstance(result, BaseModel):
        model_name: str | None = type(result).__name__
        output_root = UPath(structured_root) / model_name
        json_text = result.model_dump_json(indent=2)
    else:
        model_name = None
        output_root = UPath(structured_root) / function_name
        # Fallback: store generic JSON-serialisable data
        json_text = json.dumps(result, indent=2, default=str)

    output_root.mkdir(parents=True, exist_ok=True)

    # Preserve directory structure: compute relative path from root_dir to source file
    root_dir_path = UPath(root_dir)
    try:
        relative_source_path = upath.relative_to(root_dir_path)
    except ValueError:
        # If file is not under root_dir, use just the filename
        relative_source_path = UPath(upath.name)

    # Change extension to .json and maintain directory structure
    relative_output_path = relative_source_path.with_suffix(".json")
    output_path = output_root / relative_output_path

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json_text, encoding="utf-8")

    entry = BamlExtractionManifestEntry(
        source_hash=file_info.content_hash,
        output_path=str(relative_output_path),
        processed_at=datetime.now(timezone.utc),
    )

    logger.success(f"Wrote structured output to {output_path}")

    return str(upath), entry, model_name


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="baml_structured_extraction", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def baml_structured_extraction_flow(
    root_dir: str,
    output_dir: str,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    recursive: bool = False,
    batch_size: int = 5,
    force: bool = False,
    function_name: str,
    config_name: str = "default",
    llm: str | None = None,
) -> BamlExtractionManifest:
    """Run BAML structured extraction as a Prefect flow.

    Args:
        root_dir: Root directory to search for files (supports config variables)
        output_dir: Directory to write output files and manifest (supports config variables)
        include_patterns: List of glob patterns to include (default: ["*.md"])
        exclude_patterns: List of glob patterns to exclude (default: None)
        recursive: Search recursively in subdirectories
        batch_size: Number of files to process concurrently per batch
        force: Reprocess files even if unchanged in manifest
        function_name: BAML function name to invoke
        config_name: Configuration name from YAML config
        llm: Optional LLM identifier

    Returns:
        Updated manifest with processing results
    """

    # Resolve file list using utility function
    file_paths = resolve_files(
        root_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        recursive=recursive,
    )

    if not file_paths:
        logger.warning("No files found to process")
        # Return empty manifest
        return BamlExtractionManifest(
            function_name=function_name,
            config_name=config_name,
            llm=llm,
        )

    logger.info(f"Discovered {len(file_paths)} files to process")

    # Resolve output directory
    from genai_tk.utils.file_patterns import resolve_config_path

    resolved_output = resolve_config_path(output_dir)
    structured_root_upath = UPath(resolved_output)
    structured_root_upath.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = _find_existing_manifest(
        structured_root_upath,
        function_name=function_name,
        config_name=config_name,
        llm=llm,
    )

    if manifest is None:
        # Calculate schema fingerprint for new manifest
        try:
            schema_fp = prompt_fingerprint(function_name, config_name)
        except Exception as exc:
            logger.warning(f"Failed to compute schema fingerprint: {exc}")
            schema_fp = None

        manifest = BamlExtractionManifest(
            function_name=function_name,
            config_name=config_name,
            llm=llm,
            schema_fingerprint=schema_fp,
        )

    # Convert Path objects to UPath for processing
    files = list(_iter_markdown_files([UPath(p) for p in file_paths]))

    to_process, skipped = _prepare_files(files, manifest, force=force)

    if skipped:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        return manifest

    logger.info(f"Processing {len(to_process)} files with BAML")

    all_entries: dict[str, BamlExtractionManifestEntry] = dict(manifest.entries)
    detected_model_name: str | None = manifest.model_name

    # Process in batches to avoid oversubscribing resources.
    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                function_name=function_name,
                config_name=config_name,
                llm=llm,
                structured_root=str(structured_root_upath),
                root_dir=root_dir,
            )
            for file_info in batch
        ]

        for future in futures:
            result = future.result()  # type: ignore[misc]
            # Handle both sync and async results from Prefect tasks
            if asyncio.iscoroutine(result):
                source_path, entry, model_name = asyncio.run(result)  # type: ignore[misc]
            else:
                source_path, entry, model_name = result  # type: ignore[misc]
            all_entries[source_path] = entry
            if model_name and detected_model_name is None:
                detected_model_name = model_name

    # Determine the directory in which the manifest should be stored.
    # Prefer the detected model name when available so that the
    # directory structure is data_root/structured/<model_name>/.
    model_dir_name = detected_model_name or manifest.model_name or function_name

    if manifest_path is None:
        manifest_dir = structured_root_upath / model_dir_name
        manifest_path = manifest_dir / "manifest.json"
    else:
        manifest_dir = manifest_path.parent

    # Calculate schema fingerprint for updated manifest
    try:
        schema_fp = prompt_fingerprint(function_name, config_name)
    except Exception as exc:
        logger.warning(f"Failed to compute schema fingerprint: {exc}")
        schema_fp = manifest.schema_fingerprint  # Keep existing if calculation fails

    updated_manifest = BamlExtractionManifest(
        function_name=function_name,
        config_name=config_name,
        llm=llm,
        model_name=model_dir_name,
        schema_fingerprint=schema_fp,
        entries=all_entries,
    )

    _save_manifest(updated_manifest, manifest_path)

    logger.success(
        f"Extraction completed. {len(to_process)} files processed, {skipped} skipped. "
        f"Manifest written to {manifest_path}",
    )

    return updated_manifest


@task
async def _process_single_input_task(
    input_text: str,
    function_name: str,
    config_name: str,
    llm: str | None,
    output_dir: str | None,
    output_file: str | None,
    input_hash: str,
    force: bool,
    existing_manifest: BamlExtractionManifest | None,
) -> tuple[BaseModel | Any, str | None, str | None]:
    """Run BAML on a single text input and optionally save result as JSON.

    Returns a tuple of (result, model_name, output_path).
    If output_dir/output_file are None, result is returned without saving.
    """

    logger.info(f"Processing input with BAML function: {function_name}")

    # Check if already processed based on manifest
    if not force and existing_manifest and output_dir and output_file:
        input_key = f"input:{input_hash}"
        existing_entry = existing_manifest.entries.get(input_key)
        if existing_entry and existing_entry.source_hash == input_hash:
            # Load and return existing result
            output_path_obj = UPath(output_dir) / existing_entry.output_path
            if output_path_obj.exists():
                logger.info(f"Skipping - result already exists: {output_path_obj}")
                json_text = output_path_obj.read_text(encoding="utf-8")
                # Try to reconstruct the result, but return raw JSON if it fails
                try:
                    result_data = json.loads(json_text)
                    return result_data, existing_manifest.model_name, str(existing_entry.output_path)
                except Exception:
                    return json_text, existing_manifest.model_name, str(existing_entry.output_path)

    params: dict[str, Any] = {"__input__": input_text}
    result = await baml_invoke(function_name, params, config_name, llm)

    # If no output directory specified, just return the result
    if not output_dir or not output_file:
        model_name = type(result).__name__ if isinstance(result, BaseModel) else None
        return result, model_name, None

    # Determine model name
    if isinstance(result, BaseModel):
        model_name: str | None = type(result).__name__
        json_text = result.model_dump_json(indent=2)
    else:
        model_name = None
        json_text = json.dumps(result, indent=2, default=str)

    # Create output directory structure: output_dir/ModelName/
    output_root = UPath(output_dir)
    if model_name:
        output_root = output_root / model_name

    output_root.mkdir(parents=True, exist_ok=True)

    # Write output file
    output_path = output_root / output_file
    output_path.write_text(json_text, encoding="utf-8")

    logger.success(f"Wrote structured output to {output_path}")

    return result, model_name, str(output_path.relative_to(output_dir))


@flow(name="baml_single_input", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def baml_single_input_flow(
    input_text: str,
    function_name: str,
    *,
    config_name: str = "default",
    llm: str | None = None,
    output_dir: str | None = None,
    output_file: str | None = None,
    force: bool = False,
) -> tuple[BaseModel | Any, str | None]:
    """Run BAML on single text input as a Prefect flow.

    Args:
        input_text: Text input to process
        function_name: BAML function name to invoke
        config_name: Configuration name from YAML config
        llm: Optional LLM identifier
        output_dir: Optional output directory (supports config variables)
        output_file: Optional output filename (must end with .json)
        force: Reprocess even if result exists in manifest

    Returns:
        Tuple of (result, model_name) where result is the BAML output
    """

    # Compute input hash for caching
    input_hash = _compute_hash(input_text.encode("utf-8"))

    # Resolve output directory if provided
    resolved_output_dir: str | None = None
    if output_dir:
        from genai_tk.utils.file_patterns import resolve_config_path

        resolved_output_dir = resolve_config_path(output_dir)

    # Load existing manifest if output is configured
    existing_manifest: BamlExtractionManifest | None = None
    manifest_path: UPath | None = None

    if resolved_output_dir and output_file:
        output_root = UPath(resolved_output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        existing_manifest, manifest_path = _find_existing_manifest(
            output_root,
            function_name=function_name,
            config_name=config_name,
            llm=llm,
        )

        if existing_manifest is None:
            existing_manifest = BamlExtractionManifest(
                function_name=function_name,
                config_name=config_name,
                llm=llm,
            )

    # Process the input
    future = _process_single_input_task.submit(
        input_text=input_text,
        function_name=function_name,
        config_name=config_name,
        llm=llm,
        output_dir=resolved_output_dir,
        output_file=output_file,
        input_hash=input_hash,
        force=force,
        existing_manifest=existing_manifest,
    )

    result_tuple = future.result()  # type: ignore[misc]
    if asyncio.iscoroutine(result_tuple):
        result, model_name, relative_output_path = asyncio.run(result_tuple)  # type: ignore[misc]
    else:
        result, model_name, relative_output_path = result_tuple  # type: ignore[misc]

    # Update manifest if output was saved
    if resolved_output_dir and output_file and relative_output_path:
        input_key = f"input:{input_hash}"
        entry = BamlExtractionManifestEntry(
            source_hash=input_hash,
            output_path=relative_output_path,
            processed_at=datetime.now(timezone.utc),
        )

        if existing_manifest:
            existing_manifest.entries[input_key] = entry
            existing_manifest.model_name = model_name or existing_manifest.model_name
        else:
            existing_manifest = BamlExtractionManifest(
                function_name=function_name,
                config_name=config_name,
                llm=llm,
                model_name=model_name,
                entries={input_key: entry},
            )

        # Save manifest
        model_dir_name = model_name or function_name
        if manifest_path is None:
            manifest_dir = UPath(resolved_output_dir) / model_dir_name
            manifest_path = manifest_dir / "manifest.json"

        _save_manifest(existing_manifest, manifest_path)
        logger.success(f"Manifest updated at {manifest_path}")

    return result, model_name
