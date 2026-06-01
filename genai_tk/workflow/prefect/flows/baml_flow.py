"""Prefect-powered structured extraction for BAML outputs.

Runs BAML-based structured extraction on Markdown files and writes results as
JSON to an output directory, with a manifest to avoid duplicate processing.

Typical usage::

    uv run cli baml prefect --pathspec '**/*.md' --to ./output --function ExtractRainbow
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from genai_tk.core.factories.llm_factory import LlmFactory
from genai_tk.extra.structured.baml_util import baml_invoke, prompt_fingerprint
from genai_tk.utils.file_patterns import resolve_config_path, resolve_files
from genai_tk.utils.hashing import buffer_digest
from genai_tk.workflow.flow_cache.manifest import ManifestCache


class BamlExtractionManifestEntry(BaseModel):
    """A single BAML extraction result entry in the manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BamlExtractionManifest(BaseModel):
    """Manifest for BAML extraction results to avoid reprocessing."""

    function_name: str
    config_name: str
    llm: str | None = None
    resolved_llm: str | None = None
    model_name: str | None = None
    entries: dict[str, BamlExtractionManifestEntry] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize manifest to JSON string."""
        return super().model_dump_json(**kwargs)


def _find_existing_manifest(
    output_root: Path,
    function_name: str,
    config_name: str,
    llm: str,
) -> tuple[BamlExtractionManifest | None, Path | None]:
    """Load existing manifest from output directory if it exists.

    Returns:
        Tuple of (manifest, manifest_path). Both None if no manifest found.
    """
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_json = manifest_path.read_text(encoding="utf-8")
            data = json.loads(manifest_json)
            manifest = BamlExtractionManifest.model_validate(data)
            logger.info("Loaded existing manifest from {}", manifest_path)
            return manifest, manifest_path
        except Exception as exc:
            logger.warning("Failed to load manifest at {}: {}", manifest_path, exc)
            return None, manifest_path
    return None, None


def _save_manifest(manifest: BamlExtractionManifest, manifest_path: Path) -> None:
    """Save manifest to JSON file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = manifest.model_dump_json(indent=2)
    manifest_path.write_text(manifest_json, encoding="utf-8")


@dataclass(slots=True)
class _FileToProcess:
    path: Path
    content_hash: str
    content_text: str
    content_bytes: bytes | None = None
    is_pdf: bool = False


PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".md", ".markdown"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


def _compute_hash(content: bytes) -> str:
    return buffer_digest(content)


def _iter_supported_files(files: list[Path]) -> Iterable[Path]:
    """Yield supported files (Markdown and PDF) from a pre-resolved list."""
    for path in files:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _prepare_files(
    files: Iterable[Path],
    cache: ManifestCache,
    schema_fp: str | None,
    force: bool,
) -> tuple[list[_FileToProcess], int]:
    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = _compute_hash(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error reading {}: {}", path, exc)
            continue

        if cache.is_fresh(str(path), fingerprint=content_hash, code_version=schema_fp, force=force):
            skipped += 1
            logger.info("Skipping unchanged file: {}", path)
            continue

        is_pdf = path.suffix.lower() in PDF_EXTENSIONS
        to_process.append(
            _FileToProcess(
                path=path,
                content_hash=content_hash,
                content_text="" if is_pdf else content_bytes.decode("utf-8", errors="replace"),
                content_bytes=content_bytes if is_pdf else None,
                is_pdf=is_pdf,
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
) -> tuple[str, str, str | None]:  # (source_path, relative_output_path, model_name)

    upath = file_info.path
    logger.info("Processing file with BAML: {}", upath)

    if file_info.is_pdf and file_info.content_bytes is not None:
        import base64

        import baml_py

        pdf_b64 = base64.b64encode(file_info.content_bytes).decode("ascii")
        input_doc: Any = baml_py.Pdf.from_base64(pdf_b64)
    else:
        input_doc = file_info.content_text

    params: dict[str, Any] = {"__input__": input_doc}
    try:
        result = await baml_invoke(function_name, params, config_name, llm)
    except Exception as exc:
        from genai_tk.extra.structured.exceptions import StructuredOutputError

        if isinstance(exc, StructuredOutputError):
            logger.error("BAML error: {}", exc)
        raise

    # Determine model name and output directory. For Pydantic models we
    # use ``structured/<model_name>/`` as the root; otherwise we fall
    # back to ``structured/<function_name>/``.
    if isinstance(result, BaseModel):
        model_name: str | None = type(result).__name__
        output_root = Path(structured_root) / model_name
        json_text = result.model_dump_json(indent=2)
    else:
        model_name = None
        output_root = Path(structured_root) / function_name
        # Fallback: store generic JSON-serialisable data
        json_text = json.dumps(result, indent=2, default=str)

    output_root.mkdir(parents=True, exist_ok=True)

    # Preserve directory structure: compute relative path from root_dir to source file
    root_dir_path = Path(root_dir)
    try:
        relative_source_path = upath.relative_to(root_dir_path)
    except ValueError:
        # If file is not under root_dir, use just the filename
        relative_source_path = Path(upath.name)

    # Change extension to .json and maintain directory structure
    relative_output_path = relative_source_path.with_suffix(".json")
    output_path = output_root / relative_output_path

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json_text, encoding="utf-8")
    logger.success(f"Wrote structured output to {output_path}")

    return str(upath), str(relative_output_path), model_name


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="baml_structured_extraction", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def baml_structured_extraction_flow(
    base_dir: str,
    output_dir: str,
    *,
    pathspecs: list[str] | None = None,
    batch_size: int = 5,
    force: bool = False,
    function_name: str,
    config_name: str = "default",
    llm: str = "default",
) -> BamlExtractionManifest:
    """Run BAML structured extraction as a Prefect flow.

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        output_dir: Directory to write output files and manifest.
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).  Defaults to
            ``["**/*.md"]``.
        batch_size: Number of files to process concurrently per batch.
        force: Reprocess files even if unchanged in manifest.
        function_name: BAML function name to invoke.
        config_name: Configuration name from YAML config.
        llm: LLM identifier (``"default"`` = configured default).

    Returns:
        Updated manifest with processing results.
    """
    if pathspecs is None:
        pathspecs = ["**/*.md"]

    file_paths = resolve_files(base_dir, pathspecs=pathspecs)

    if not file_paths:
        logger.warning("No files found to process")
        return BamlExtractionManifest(
            function_name=function_name,
            config_name=config_name,
            llm=llm,
        )

    logger.info("Discovered {} files to process", len(file_paths))

    resolved_llm: str | None = None
    if llm and llm != "default":
        resolved_llm = LlmFactory(llm=llm).get_id()
        logger.info("Resolved BAML LLM override '{}' -> '{}'", llm, resolved_llm)

    resolved_output = resolve_config_path(output_dir)
    structured_root_upath = Path(resolved_output)
    structured_root_upath.mkdir(parents=True, exist_ok=True)

    # Compute schema fingerprint once (used as code_version for cache freshness)
    schema_fp: str | None = None
    try:
        schema_fp = prompt_fingerprint(function_name, config_name)
    except Exception as exc:
        logger.debug("Schema fingerprint unavailable (multimodal types not supported by fingerprinter): {}", exc)

    manifest_path = structured_root_upath / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    files = list(_iter_supported_files([Path(p) for p in file_paths]))
    to_process, skipped = _prepare_files(files, cache, schema_fp=schema_fp, force=force)

    if skipped:
        logger.info("Skipped {} unchanged files based on manifest", skipped)

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        manifest = BamlExtractionManifest(
            function_name=function_name,
            config_name=config_name,
            llm=llm,
            resolved_llm=resolved_llm,
            entries={
                k: BamlExtractionManifestEntry(
                    source_hash=rec.fingerprint,
                    output_path=rec.outputs.get("output_path", ""),
                    processed_at=rec.processed_at,
                )
                for k, rec in cache.records.items()
            },
        )
        return manifest

    logger.info("Processing {} files with BAML", len(to_process))

    detected_model_name: str | None = None

    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                function_name=function_name,
                config_name=config_name,
                llm=llm,
                structured_root=str(structured_root_upath),
                root_dir=base_dir,
            )
            for file_info in batch
        ]

        for future in futures:
            result = future.result()  # type: ignore[misc]
            if asyncio.iscoroutine(result):
                source_path, relative_output_path, model_name = asyncio.run(result)  # type: ignore[misc]
            else:
                source_path, relative_output_path, model_name = result  # type: ignore[misc]

            file_hash = next((f.content_hash for f in to_process if str(f.path) == source_path), "")
            cache.record_success(
                key=source_path,
                fingerprint=file_hash,
                outputs={"output_path": relative_output_path},
                code_version=schema_fp,
            )
            if model_name and detected_model_name is None:
                detected_model_name = model_name

    cache.save(manifest_path)
    logger.success(f"Extraction completed. {len(to_process)} files processed, {skipped} skipped.")

    manifest = BamlExtractionManifest(
        function_name=function_name,
        config_name=config_name,
        llm=llm,
        resolved_llm=resolved_llm,
        model_name=detected_model_name,
        entries={
            k: BamlExtractionManifestEntry(
                source_hash=rec.fingerprint,
                output_path=rec.outputs.get("output_path", ""),
                processed_at=rec.processed_at,
            )
            for k, rec in cache.records.items()
        },
    )
    return manifest


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

    logger.info("Processing input with BAML function: {}", function_name)

    # Check if already processed based on manifest
    if not force and existing_manifest and output_dir and output_file:
        input_key = f"input:{input_hash}"
        existing_entry = existing_manifest.entries.get(input_key)
        if existing_entry and existing_entry.source_hash == input_hash:
            # Load and return existing result
            output_path_obj = Path(output_dir) / existing_entry.output_path
            if output_path_obj.exists():
                logger.info("Skipping - result already exists: {}", output_path_obj)
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
    output_root = Path(output_dir)
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
    llm: str = "default",
    output_dir: str | None = None,
    output_file: str | None = None,
    force: bool = False,
) -> tuple[BaseModel | Any, str | None]:
    """Run BAML on single text input as a Prefect flow.

    Args:
        input_text: Text input to process
        function_name: BAML function name to invoke
        config_name: Configuration name from YAML config
        llm: LLM identifier (use "default" to use configured default)
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
    manifest_path: Path | None = None

    if resolved_output_dir and output_file:
        output_root = Path(resolved_output_dir)
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
            manifest_dir = Path(resolved_output_dir) / model_dir_name
            manifest_path = manifest_dir / "manifest.json"

        _save_manifest(existing_manifest, manifest_path)
        logger.success(f"Manifest updated at {manifest_path}")

    return result, model_name
