"""Prefect-powered PII anonymization for text files.

Reads text files, detects and replaces PII using Presidio + Faker, and writes
anonymized copies to an output directory.

The core anonymization logic lives in :mod:`genai_tk.workflow.anonymization.core`,
shared with :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware`.
This ensures identical behaviour whether PII scrubbing happens at ETL time (this flow)
or at agent runtime (the middleware).

Optionally a ``.mapping.json`` sidecar file is written alongside each output so
that RAG answers can be *deanonymized* later if needed.

Typical usage::

    uv run cli workflow run anonymize_docs --dry-run
    uv run cli workflow run anonymize_docs
    uv run cli workflow run anonymize_and_ingest_docs
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner  # type: ignore[attr-defined]

from genai_tk.config_mgmt.file_patterns import resolve_files
from genai_tk.utils.hashing import buffer_digest
from genai_tk.workflow.anonymization.core import (
    AnonymizationConfig,
    anonymize_text,
)
from genai_tk.workflow.anonymization.presidio_detector import (
    PresidioDetector,
    PresidioDetectorConfig,
)
from genai_tk.workflow.flow_cache.manifest import ManifestCache

# ---------------------------------------------------------------------------
# Prefect task
# ---------------------------------------------------------------------------


@task(log_prints=False)
def anonymize_file_task(
    source_path: str,
    output_dir: str,
    root_dir: str,
    config: AnonymizationConfig,
    save_mapping: bool = False,
) -> tuple[str, str, str | None] | None:  # (output_path, source_hash, mapping_path)
    """Anonymize a single text file and write the result to *output_dir*.

    Args:
        source_path: Absolute path to the source file.
        output_dir: Root output directory (directory structure is preserved).
        root_dir: Root source directory used to compute relative output paths.
        config: :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationConfig`
            controlling which entities to detect and Faker settings.
        save_mapping: If ``True``, write a ``<filename>.mapping.json`` sidecar file next to
            the anonymized output so that answers can be deanonymized later.

    Returns:
        A :class:`AnonymizeManifestEntry` on success, ``None`` on failure.
    """
    from faker import Faker

    upath = Path(source_path)
    output_upath = Path(output_dir)
    root_upath = Path(root_dir)

    try:
        text = upath.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read {}: {}", upath, exc)
        return None

    detector = PresidioDetector(config=config.detector)
    if config.faker_seed is not None:
        Faker.seed(config.faker_seed)
    faker = Faker(locale=config.faker_locales)

    anonymized, mapping = anonymize_text(text, detector=detector, faker=faker)

    # Preserve input directory structure in output
    try:
        relative_path = upath.relative_to(root_upath)
    except ValueError:
        relative_path = Path(upath.name)

    output_path = output_upath / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(anonymized, encoding="utf-8")

    mapping_path: str | None = None
    if save_mapping and mapping:
        mp = output_path.parent / (output_path.name + ".mapping.json")
        mp.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        mapping_path = str(
            (output_upath / relative_path.parent / (relative_path.name + ".mapping.json")).relative_to(output_upath)
        )

    source_hash = buffer_digest(upath.read_bytes())
    logger.success("Anonymized {} entities in {}", len(mapping), upath.name)

    return str(relative_path), source_hash, mapping_path


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(
    name="Anonymize Files",
    description="Detect and replace PII in text files using Presidio + Faker",
    task_runner=ThreadPoolTaskRunner(max_workers=1),  # type: ignore[call-arg]
)
def anonymize_files_flow(
    base_dir: str,
    output_dir: str,
    *,
    pathspecs: list[str] | None = None,
    batch_size: int = 10,
    force: bool = False,
    save_mapping: bool = False,
    analyzed_fields: list[str] | None = None,
    faker_seed: int | None = 42,
    fuzzy_deanonymize: bool = True,
) -> ManifestCache:
    """Anonymize PII in text files and write cleaned copies to *output_dir*.

    Reuses the same :func:`~genai_tk.agents.langchain.middleware.anonymization_middleware.anonymize_text`
    function as :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware`,
    so entity detection and Faker replacement are identical whether anonymization happens at
    ETL time (this flow) or at agent runtime (the middleware).

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        output_dir: Directory to write anonymized files and manifest.
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).
            Defaults to ``["**/*.txt", "**/*.md"]``.
        batch_size: Number of files to process concurrently per batch.
        force: Reprocess files even if unchanged in the manifest.
        save_mapping: Write per-file ``.mapping.json`` sidecar files for later deanonymization.
        analyzed_fields: Presidio entity types to detect.  Uses the default set when ``None``.
        faker_seed: Seed for deterministic Faker output (``None`` = random each run).
        fuzzy_deanonymize: Stored in the config for downstream agent middleware reuse.

    Returns:
        Updated :class:`AnonymizeManifest` with entries for all processed files.
    """
    if pathspecs is None:
        pathspecs = ["**/*.txt", "**/*.md"]

    config = AnonymizationConfig(
        detector=PresidioDetectorConfig(
            analyzed_fields=analyzed_fields or PresidioDetectorConfig().analyzed_fields,
        ),
        faker_seed=faker_seed,
        fuzzy_deanonymize=fuzzy_deanonymize,
    )

    # Load manifest for incremental processing
    output_upath = Path(output_dir)
    manifest_path = output_upath / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    files = resolve_files(base_dir, pathspecs=pathspecs)
    logger.info("Found {} files matching patterns in '{}'", len(files), base_dir)

    if not files:
        logger.warning("No files found to anonymize")
        return cache

    files_to_process: list[Path] = []
    skipped = 0
    for f in files:
        try:
            content_hash = buffer_digest(Path(f).read_bytes())
        except Exception as exc:
            logger.error("Cannot read {}: {}", f, exc)
            continue
        if cache.is_fresh(str(f), fingerprint=content_hash, force=force):
            skipped += 1
            continue
        files_to_process.append(Path(f))

    logger.info("Processing {} files, skipping {} unchanged", len(files_to_process), skipped)

    if not files_to_process:
        return cache

    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info("Batch {}/{} ({} files)", i // batch_size + 1, -(-len(files_to_process) // batch_size), len(batch))

        futures = [
            anonymize_file_task.submit(
                source_path=str(f),
                output_dir=output_dir,
                root_dir=base_dir,
                config=config,
                save_mapping=save_mapping,
            )
            for f in batch
        ]

        for f, future in zip(batch, futures, strict=False):
            try:
                result = future.result()
                if result is not None:
                    output_path, source_hash, mapping_path = result
                    outputs: dict = {"output_path": output_path}
                    if mapping_path:
                        outputs["mapping_path"] = mapping_path
                    cache.record_success(key=str(f), fingerprint=source_hash, outputs=outputs)
            except Exception as exc:
                logger.error("Failed to process {}: {}", f, exc)

    output_upath.mkdir(parents=True, exist_ok=True)
    cache.save(manifest_path)
    logger.success(
        "Anonymization complete: {} processed, {} skipped",
        len(files_to_process),
        skipped,
    )
    return cache
