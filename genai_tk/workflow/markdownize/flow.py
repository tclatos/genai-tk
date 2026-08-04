"""Prefect-powered Markdown conversion driven by a single ``markdownize`` profile.

One call — :func:`markdownize_flow` — converts a mix of directories, ``.zip``
archives, and individual files (Office / PDF / image / pre-existing Markdown)
to Markdown. Callers pick a **profile** (``fast`` / ``medium`` / ``best`` —
see :mod:`genai_tk.workflow.markdownize.config`) instead of wiring
low-level converter flags; the profile decides, per source-document family, how
each file becomes Markdown:

- ``.zip`` archives are extracted into ``cache_dir/unzipped/`` first.
- Pre-existing ``.md`` / ``.markdown`` files are copied through unchanged.
- PowerPoint / Word can go straight through ``markitdown`` or ``via_pdf``
  (LibreOffice → PDF → OCR, staged under ``cache_dir/pdf/``). The ``via_pdf``
  hop is an internal detail.
- Spreadsheets add a deterministic ``messy_xls_parser`` option (see
  :mod:`genai_tk.workflow.markdownize.excel`): merged title/banner cells and
  grouped multi-row headers are recovered, stacked and side-by-side tables in
  the same sheet are split apart, free-text/legend blocks are kept as prose,
  and dates/percentages/numbers are formatted from the cell's actual value.
- Every PDF — native *and* the ones produced by ``via_pdf`` — is turned into
  Markdown by the profile's ``pdf_converter`` (``mistral`` / ``markitdown`` /
  ``edgeparse``). When it is ``mistral``, *all* PDFs are sent in a single Mistral
  batch job, which is cheaper than per-file OCR.

Typical usage::

    uv run cli workflow run markdownize --set sources=./docs --set md_output_dir=./md

Programmatic::

    from genai_tk.workflow.markdownize import markdownize_flow

    markdownize_flow(sources="./docs", md_output_dir="./md", profile="medium")
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]

from genai_tk.config_mgmt.file_patterns import resolve_config_path
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.force import ForceStage, stage_active
from genai_tk.workflow.markdownize.config import MarkdownizeProfile, get_markdownize_profile
from genai_tk.workflow.markdownize.converters import _convert_text
from genai_tk.workflow.markdownize.manifest import MarkdownizeManifest, MarkdownizeManifestEntry
from genai_tk.workflow.markdownize.mistral import _ocr_pdfs_with_mistral
from genai_tk.workflow.markdownize.routing import (
    DIRECT_MARKITDOWN_EXTS,
    DOC_EXTS,
    EXCEL_EXTS,
    IMAGE_EXTS,
    MD_EXTS,
    PPT_EXTS,
    _is_markdownize_compatible,
    _output_paths,
    _prepare_files,
    _route_for,
    _write_markdown,
)
from genai_tk.workflow.prefect.flows.office2pdf_flow import (
    _convert_with_libreoffice,
    ensure_libreoffice_available,
)
from genai_tk.workflow.sources import resolve_sources


@task(log_prints=False)
def _libreoffice_task(src: str, pdf_root: str, root_dir: str) -> tuple[str, str]:
    """Convert one Office file to PDF, preserving structure. Returns (src, pdf_path)."""
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
    route: str,
    out_abs: str,
    rel_out: str,
    pdf_converter: str,
) -> tuple[str, str]:
    """Convert one file (``convert_src``) to Markdown and write it. Returns (original_src, rel_out)."""
    text = _convert_text(Path(convert_src), route, pdf_converter)
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
    """Build a :class:`MarkdownizeManifest` view of the current cache records."""
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


def _publish_summary(manifest: MarkdownizeManifest, processed_keys: set[str], skipped: int) -> None:
    """Publish a best-effort Prefect artifact summarising the conversion run."""
    try:
        from prefect.artifacts import create_markdown_artifact

        lines = [
            "# Markdownize Summary",
            "",
            f"**Processed:** {len(processed_keys)}  |  **Skipped (cached):** {skipped}",
            "",
        ]
        if processed_keys:
            lines += ["## Converted files", "", "| Source file | Output |", "|-------------|--------|"] + [
                f"| `{Path(k).name}` | `{v.output_path}` |" for k, v in manifest.entries.items() if k in processed_keys
            ]
        cached = [k for k in manifest.entries if k not in processed_keys]
        if cached:
            lines += ["", "## Cached (skipped)", ""] + [f"- `{Path(k).name}`" for k in cached]
        create_markdown_artifact("\n".join(lines), key="markdownize-summary")
    except Exception:
        pass  # Artifacts are best-effort; never block the return value.


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
    """Convert directories / ``.zip`` archives / files to Markdown using a markdownize profile.

    Args:
        sources: One source, or a list of sources — directories, ``.zip``
            archives, or individual files. Supports ``${paths.*}`` config vars.
            ``.zip`` archives are extracted into ``cache_dir/unzipped/``.
            Pre-existing ``.md`` / ``.markdown`` files are copied through unchanged.
        md_output_dir: Directory to write the final Markdown files.
        cache_dir: Directory for intermediates (unzipped archives, ``via_pdf``
            PDFs, the manifest). Defaults to ``<md_output_dir>/.cache``.
        profile: A :class:`MarkdownizeProfile` or the name of one (``fast`` /
            ``medium`` / ``best`` / ``default``, or a key configured under
            ``markdownize_profiles``).  The profile decides, per source-document
            family, whether to convert directly (markitdown / messy_xls_parser) or
            ``via_pdf`` (LibreOffice → PDF), and which ``pdf_converter`` turns
            every PDF into Markdown.  When ``pdf_converter`` is ``mistral`` all
            PDFs are sent in a single, cheaper Mistral batch job.
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).  Defaults to
            all supported document extensions.
        batch_size: Number of files converted concurrently per batch.
        force_stage: One of ``unzip``, ``pdf``, ``md``, or ``all`` (see
            :mod:`genai_tk.workflow.force`). ``unzip`` re-extracts zip archives
            even if cached; ``pdf``/``md``/``all`` reprocess every file even if
            unchanged in the manifest.
        already_processed: Optional callback taking a file's content hash and
            returning True when a downstream store already holds its converted
            output, so the file can be skipped without a manifest entry.

    Returns:
        Updated manifest with processing results.
    """
    from genai_tk.utils.prefect_logging import install_loguru_prefect_bridge

    install_loguru_prefect_bridge()

    resolved_profile = profile if isinstance(profile, MarkdownizeProfile) else get_markdownize_profile(profile)
    logger.info(f"Markdownize profile: {resolved_profile.fingerprint()}")

    if pathspecs is None:
        pathspecs = [
            f"**/*{ext}"
            for ext in sorted(
                PPT_EXTS | DOC_EXTS | EXCEL_EXTS | IMAGE_EXTS | DIRECT_MARKITDOWN_EXTS | MD_EXTS | {".pdf"}
            )
        ]

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

    logger.info(f"Discovered {len(resolved_files)} files to process")

    manifest_path = cache_root / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    code_version = resolved_profile.fingerprint()
    compatible = [rf for rf in resolved_files if _is_markdownize_compatible(rf.path)]
    to_process, skipped = _prepare_files(
        compatible, cache, force=force_convert, already_processed=already_processed, code_version=code_version
    )

    if skipped:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        manifest = _manifest_from_cache(cache)
        _publish_summary(manifest, processed_keys=set(), skipped=skipped)
        return manifest

    logger.info(f"Processing {len(to_process)} files")

    routes = {str(fi.path): _route_for(fi.path, resolved_profile) for fi in to_process}
    results: list[tuple[str, str]] = []

    # 1. LibreOffice-convert every 'via_pdf' file to an intermediate PDF.
    via_pdf_files = [fi for fi in to_process if routes[str(fi.path)] == "via_pdf"]
    pdf_source: dict[str, str] = {}
    if via_pdf_files:
        ensure_libreoffice_available()
        pdf_root = cache_root / "pdf"
        logger.info(f"Converting {len(via_pdf_files)} document(s) to PDF via LibreOffice")
        for batch in _chunked(via_pdf_files, batch_size):
            futures = [_libreoffice_task.submit(str(fi.path), str(pdf_root), str(fi.root)) for fi in batch]
            for future in futures:
                src, pdf = future.result()  # type: ignore[misc]
                pdf_source[src] = pdf

    # 2. Native PDFs join the same PDF pipeline as-is.
    for fi in to_process:
        if routes[str(fi.path)] == "pdf":
            pdf_source[str(fi.path)] = str(fi.path)

    # 3. Turn every PDF (native + via_pdf) into Markdown with the profile's pdf_converter.
    pdf_items = [(fi, pdf_source[str(fi.path)]) for fi in to_process if str(fi.path) in pdf_source]
    if pdf_items:
        if resolved_profile.pdf_converter == "mistral":
            results.extend(_ocr_pdfs_with_mistral(pdf_items, output_upath))
        else:
            for batch in _chunked(pdf_items, batch_size):
                futures = []
                for fi, pdf in batch:
                    rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath)
                    futures.append(
                        _convert_file_task.submit(
                            str(fi.path), pdf, "pdf", str(out_abs), str(rel_out), resolved_profile.pdf_converter
                        )
                    )
                results.extend(f.result() for f in futures)  # type: ignore[misc]

    # 4. Direct conversions: messy_xls_parser spreadsheets + markitdown for everything else.
    direct_files = [fi for fi in to_process if routes[str(fi.path)] in ("messy_xls_parser", "markitdown")]
    for batch in _chunked(direct_files, batch_size):
        futures = []
        for fi in batch:
            rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath)
            futures.append(
                _convert_file_task.submit(
                    str(fi.path),
                    str(fi.path),
                    routes[str(fi.path)],
                    str(out_abs),
                    str(rel_out),
                    resolved_profile.pdf_converter,
                )
            )
        results.extend(f.result() for f in futures)  # type: ignore[misc]

    # 5. Pre-existing Markdown: copied through unchanged.
    copy_files = [fi for fi in to_process if routes[str(fi.path)] == "copy"]
    for batch in _chunked(copy_files, batch_size):
        futures = []
        for fi in batch:
            rel_out, out_abs = _output_paths(fi.path, fi.root, output_upath, route="copy")
            futures.append(_copy_file_task.submit(str(fi.path), str(out_abs), str(rel_out)))
        results.extend(f.result() for f in futures)  # type: ignore[misc]

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
    _publish_summary(manifest, processed_keys={src for src, _ in results}, skipped=skipped)
    return manifest
