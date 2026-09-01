"""Prefect-powered Markdown conversion driven by markdownize profiles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from pathlib import Path

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]

from genai_tk.config_mgmt.file_patterns import resolve_config_path
from genai_tk.extra.markdownize.factory import ConverterFactory
from genai_tk.extra.markdownize.selector import MarkdownizeProfile
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.force import ForceStage, stage_active
from genai_tk.workflow.markdownize.config import get_markdownize_profile
from genai_tk.workflow.markdownize.manifest import MarkdownizeManifest, MarkdownizeManifestEntry
from genai_tk.workflow.markdownize.routing import (
    ALL_DOCUMENT_EXTS,
    _is_markdownize_compatible,
    _output_paths,
    _prepare_files,
    _write_markdown,
)
from genai_tk.workflow.prefect.flows.office2pdf_flow import (
    _convert_with_libreoffice,
    ensure_libreoffice_available,
)
from genai_tk.workflow.sources import resolve_sources


@task(log_prints=False)
def _libreoffice_task(src: str, pdf_root: str, root_dir: str) -> tuple[str, str]:
    """Convert one Office file to PDF via LibreOffice. Returns (src, pdf_path)."""
    src_path = Path(src)
    try:
        rel = src_path.relative_to(Path(root_dir))
    except ValueError:
        rel = Path(src_path.name)
    pdf = _convert_with_libreoffice(src_path, Path(pdf_root) / rel.parent)
    return src, str(pdf)


@task(log_prints=False)
def _convert_file_task(
    original_src: str,
    convert_src: str,
    converter_name: str,
    out_abs: str,
    rel_out: str,
) -> tuple[str, str]:
    """Convert one file to Markdown using ConverterFactory."""
    converter = ConverterFactory.create(converter_name)
    text = asyncio.run(converter.convert(Path(convert_src)))
    _write_markdown(Path(out_abs), Path(original_src), text)
    return original_src, rel_out


@task(log_prints=False)
def _copy_file_task(original_src: str, out_abs: str, rel_out: str) -> tuple[str, str]:
    """Copy a pre-existing Markdown file into the output tree unchanged."""
    import shutil

    out_path = Path(out_abs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original_src, out_path)
    logger.success(f"Copied markdown to {out_path}")
    return original_src, rel_out


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _manifest_from_cache(cache: ManifestCache) -> MarkdownizeManifest:
    """Build a MarkdownizeManifest view of the current cache records."""
    return MarkdownizeManifest(
        entries={
            k: MarkdownizeManifestEntry(
                source_hash=rec.fingerprint,
                output_path=rec.outputs.get("output_path", ""),
                processed_at=rec.processed_at,
            )
            for k, rec in cache.records.items()
        }
    )


@flow(name="markdownize", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def markdownize_flow(
    sources: str | list[str],
    md_output_dir: str,
    *,
    cache_dir: str | None = None,
    profile: str | MarkdownizeProfile = "default",
    pathspecs: list[str] | None = None,
    batch_size: int = 5,
    force_stage: str | ForceStage | None = None,
    already_processed: Callable[[str], bool] | None = None,
) -> MarkdownizeManifest:
    """Convert directories, archives, or files to Markdown using a MarkdownizeProfile.

    Args:
        sources: Directories, archives, or files to convert.
        md_output_dir: Directory to write final Markdown files.
        cache_dir: Cache directory for manifest and intermediates.
        profile: Profile instance or name from configuration.
        pathspecs: File matching pathspecs.
        batch_size: Concurrency batch size.
        force_stage: Stage to force re-execution.
        already_processed: Optional callback to skip cached store files.

    Returns:
        Manifest recording conversion outcomes.
    """
    from genai_tk.utils.prefect_logging import install_loguru_prefect_bridge

    install_loguru_prefect_bridge()

    resolved_profile = profile if isinstance(profile, MarkdownizeProfile) else get_markdownize_profile(profile)
    logger.info(f"Markdownize profile: {resolved_profile.fingerprint()}")

    if pathspecs is None:
        pathspecs = [f"**/*{ext}" for ext in sorted(ALL_DOCUMENT_EXTS)]

    resolved_output = resolve_config_path(md_output_dir)
    output_upath = Path(resolved_output)
    output_upath.mkdir(parents=True, exist_ok=True)

    cache_root = Path(resolve_config_path(cache_dir)) if cache_dir else output_upath / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    force_unzip = stage_active(force_stage, ForceStage.unzip)
    force_convert = stage_active(force_stage, ForceStage.md)

    resolved_files = resolve_sources(sources, cache_dir=cache_root, pathspecs=pathspecs, force_unzip=force_unzip)

    if not resolved_files:
        logger.warning("No files found to process")
        return MarkdownizeManifest()

    manifest_path = cache_root / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    code_version = resolved_profile.fingerprint()
    compatible = [rf for rf in resolved_files if _is_markdownize_compatible(rf.path)]
    to_process, skipped = _prepare_files(
        compatible, cache, force=force_convert, already_processed=already_processed, code_version=code_version
    )

    if not to_process:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")
        return _manifest_from_cache(cache)

    logger.info(f"Processing {len(to_process)} files")

    routes = {str(fi.path): resolved_profile.select_route(fi.path) for fi in to_process}
    results: list[tuple[str, str]] = []

    # 1. LibreOffice-convert 'via_pdf' files
    via_pdf_files = [fi for fi in to_process if routes[str(fi.path)] == "via_pdf"]
    pdf_source: dict[str, str] = {}
    if via_pdf_files:
        ensure_libreoffice_available()
        pdf_root = cache_root / "pdf"
        for batch in _chunked(via_pdf_files, batch_size):
            futures = [_libreoffice_task.submit(str(fi.path), str(pdf_root), str(fi.root)) for fi in batch]
            for future in futures:
                src, pdf = future.result()  # type: ignore[misc]
                pdf_source[src] = pdf

    # 2. Convert PDFs (intermediate via_pdf or native)
    pdf_converter_name = resolved_profile.select_route(Path("sample.pdf"))
    if pdf_converter_name == "via_pdf":
        pdf_converter_name = "mistral_ocr"

    pdf_items: list[tuple] = []
    for fi in to_process:
        if str(fi.path) in pdf_source:
            pdf_items.append((fi, pdf_source[str(fi.path)]))
        elif fi.path.suffix.lower() == ".pdf":
            pdf_items.append((fi, str(fi.path)))

    if pdf_items:
        if pdf_converter_name in ("mistral", "mistral_ocr"):
            converter = ConverterFactory.create("mistral_ocr")
            pdf_paths = [Path(p) for _, p in pdf_items]
            try:
                texts = asyncio.run(converter.batch_convert(pdf_paths))
            except Exception as e:
                logger.warning(f"Mistral batch OCR failed ({e}); falling back to anydoc.")
                texts = {}
                try:
                    anydoc_fb = ConverterFactory.create("anydoc")
                    for _, p in pdf_items:
                        try:
                            t = asyncio.run(anydoc_fb.convert(Path(p)))
                            if t and t.strip():
                                texts[str(p)] = t
                        except Exception:
                            pass
                except Exception:
                    pass

                # Final fallback for any missing PDFs
                missing = [p for _, p in pdf_items if str(p) not in texts]
                if missing:
                    logger.warning("Falling back to markitdown for {} files", len(missing))
                    fallback = ConverterFactory.create("markitdown")
                    for p in missing:
                        texts[str(p)] = asyncio.run(fallback.convert(Path(p)))
            for file_info, pdf in pdf_items:
                rel_out, out_abs = _output_paths(file_info.path, file_info.root, output_upath)
                text = texts.get(str(pdf), "")
                _write_markdown(out_abs, file_info.path, text)
                results.append((str(file_info.path), str(rel_out)))
        else:
            for batch in _chunked(pdf_items, batch_size):
                futures = []
                for fi, pdf in batch:
                    rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath)
                    futures.append(
                        _convert_file_task.submit(str(fi.path), pdf, pdf_converter_name, str(out_abs), str(rel_out))
                    )
                results.extend(f.result() for f in futures)  # type: ignore[misc]

    # 3. Direct conversions for all other non-PDF, non-via_pdf, non-copy files
    pdf_processed_sources = {str(fi.path) for fi, _ in pdf_items}
    direct_files = [
        fi for fi in to_process if routes[str(fi.path)] != "copy" and str(fi.path) not in pdf_processed_sources
    ]
    for batch in _chunked(direct_files, batch_size):
        futures = []
        for fi in batch:
            rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath)
            converter_name = routes[str(fi.path)]
            if converter_name == "messy_xls_parser":
                converter_name = "messy_xls"
            futures.append(
                _convert_file_task.submit(str(fi.path), str(fi.path), converter_name, str(out_abs), str(rel_out))
            )
        results.extend(f.result() for f in futures)  # type: ignore[misc]

    # 4. Copy pre-existing Markdown files
    copy_files = [fi for fi in to_process if routes[str(fi.path)] == "copy"]
    for batch in _chunked(copy_files, batch_size):
        futures = []
        for fi in batch:
            rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath, route="copy")
            futures.append(_copy_file_task.submit(str(fi.path), str(out_abs), str(rel_out)))
        results.extend(f.result() for f in futures)  # type: ignore[misc]

    # Record manifest
    for source_path, relative_output_path in results:
        file_hash = next((f.content_hash for f in to_process if str(f.path) == source_path), "")
        cache.record_success(
            key=source_path,
            fingerprint=file_hash,
            code_version=code_version,
            outputs={"output_path": relative_output_path},
        )

    cache.save(manifest_path)
    logger.success(f"Conversion completed. {len(results)} files processed, {skipped} skipped.")

    manifest = _manifest_from_cache(cache)
    return manifest
