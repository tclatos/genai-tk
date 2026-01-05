"""Prefect-powered structured extraction for BAML outputs.

This module defines a Prefect flow that runs BAML-based structured
extraction on Markdown files, writes results as JSON files to a directory
rooted in the application configuration, and maintains a manifest file to
avoid duplicate processing.

Typical usage from the CLI is similar to:

```bash
uv run cli baml prefect-extract /path/to/docs --function ExtractRainbow --force
```
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from pydantic import BaseModel, Field
from upath import UPath

from genai_tk.extra.structured.baml_util import baml_invoke
from genai_tk.utils.config_mngr import global_config


class BamlExtractionManifestEntry(BaseModel):
    """Single processed file entry in the extraction manifest."""

    source_path: str
    source_hash: str
    output_path: str
    processed_at: datetime


class BamlExtractionManifest(BaseModel):
    """Manifest of processed files for a given BAML function."""

    function_name: str
    config_name: str
    llm: str | None = None
    model_name: str | None = None
    entries: dict[str, BamlExtractionManifestEntry] = Field(default_factory=dict)


@dataclass(slots=True)
class _FileToProcess:
    path: UPath
    content_hash: str
    content_text: str


def _encode_path_to_filename(source_path: str) -> str:
    """Encode source path into a filesystem-friendly JSON file name."""

    safe = source_path.replace("\\", "_").replace("/", "_")
    return f"{safe}.json"


def _compute_hash(content: bytes) -> str:
    return sha256(content).hexdigest()


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
        if (
            manifest.function_name == function_name
            and manifest.config_name == config_name
            and manifest.llm == llm
        ):
            return manifest, candidate

    return None, None


def _iter_markdown_files(root: UPath, recursive: bool) -> Iterable[UPath]:
    pattern = "*.[mM][dD]"
    if root.is_file() and root.suffix.lower() in {".md", ".markdown"}:
        yield root
    elif root.is_dir():
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        yield from iterator


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
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {path}: {exc}")
            continue

        content_hash = _compute_hash(content_bytes)
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
) -> tuple[BamlExtractionManifestEntry, str | None]:
    """Run BAML on a single file and persist result as JSON.

    Returns a tuple of the new manifest entry and the detected model name.
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

    output_filename = _encode_path_to_filename(str(upath))
    output_path = output_root / output_filename

    output_path.write_text(json_text, encoding="utf-8")

    entry = BamlExtractionManifestEntry(
        source_path=str(upath),
        source_hash=file_info.content_hash,
        output_path=str(output_path),
        processed_at=datetime.now(timezone.utc),
    )

    logger.success(f"Wrote structured output to {output_path}")

    return entry, model_name


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="baml_structured_extraction", task_runner=ConcurrentTaskRunner())
def baml_structured_extraction_flow(
    source: str,
    *,
    recursive: bool,
    batch_size: int,
    force: bool,
    function_name: str,
    config_name: str,
    llm: str | None = None,
) -> BamlExtractionManifest:
    """Run BAML structured extraction as a Prefect flow.

    The flow discovers Markdown files under ``source``, skips files whose
    content hash is unchanged according to the manifest, processes the
    remaining files in parallel batches, and updates the manifest.
    """

    cfg = global_config()
    data_root = cfg.get_dir_path("paths.data_root", create_if_not_exists=True)
    structured_root = data_root / "structured"

    root = UPath(source)
    if not root.exists():
        msg = f"Source path does not exist: {root}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    structured_root_upath = UPath(structured_root)
    manifest, manifest_path = _find_existing_manifest(
        structured_root_upath,
        function_name=function_name,
        config_name=config_name,
        llm=llm,
    )

    # If no manifest exists yet, start with an empty one so that
    # _prepare_files builds the list of files to process without any
    # deduplication on the first run.
    if manifest is None:
        manifest = BamlExtractionManifest(
            function_name=function_name,
            config_name=config_name,
            llm=llm,
        )

    files = list(_iter_markdown_files(root, recursive=recursive))
    if not files:
        logger.warning("No Markdown files found to process")
        return manifest

    logger.info(f"Discovered {len(files)} Markdown files under {root}")

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
                structured_root=str(structured_root),
            )
            for file_info in batch
        ]

        for future in futures:
            entry, model_name = future.result()
            all_entries[entry.source_path] = entry
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

    updated_manifest = BamlExtractionManifest(
        function_name=function_name,
        config_name=config_name,
        llm=llm,
        model_name=model_dir_name,
        entries=all_entries,
    )

    _save_manifest(updated_manifest, manifest_path)

    logger.success(
        f"Extraction completed. {len(to_process)} files processed, {skipped} skipped. "
        f"Manifest written to {manifest_path}",
    )

    return updated_manifest
