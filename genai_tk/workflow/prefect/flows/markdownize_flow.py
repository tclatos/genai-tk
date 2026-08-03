"""Prefect-powered Markdown conversion driven by a single ``markdownize`` profile.

One call — :func:`markdownize_flow` — converts a directory of mixed Office / PDF /
image documents to Markdown. Callers pick a **profile** (``fast`` / ``medium`` /
``best`` — see :mod:`genai_tk.config_mgmt.markdownize_config`) instead of wiring
low-level converter flags; the profile decides, per source-document family, how
each file becomes Markdown:

- PowerPoint / Word can go straight through ``markitdown`` or ``via_pdf``
  (LibreOffice → PDF → OCR). The ``via_pdf`` hop is an internal detail.
- Spreadsheets add a deterministic ``md_parser`` option (``md-spreadsheet-parser``:
  empty rows/columns dropped, no ``NaN``, merged headers forward-filled).
- Every PDF — native *and* the ones produced by ``via_pdf`` — is turned into
  Markdown by the profile's ``pdf_converter`` (``mistral`` / ``markitdown`` /
  ``edgeparse``). When it is ``mistral``, *all* PDFs are sent in a single Mistral
  batch job, which is cheaper than per-file OCR.

Typical usage::

    uv run cli workflow run markdownize --set base_dir=./docs --set output_dir=./md

Programmatic::

    from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow

    markdownize_flow(base_dir="./docs", output_dir="./md", profile="medium")
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from genai_tk.config_mgmt.file_patterns import resolve_config_path, resolve_files
from genai_tk.config_mgmt.markdownize_config import MarkdownizeProfile, get_markdownize_profile
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.prefect.flows.office2pdf_flow import (
    _convert_with_libreoffice,
    ensure_libreoffice_available,
)

# Source-document families and how markdownize_flow routes each suffix.
PPT_EXTS = {".ppt", ".pptx", ".odp"}
DOC_EXTS = {".doc", ".docx", ".odt", ".rtf"}
EXCEL_EXTS = {".xls", ".xlsx", ".ods"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".gif", ".bmp"}
DIRECT_MARKITDOWN_EXTS = {".html", ".htm", ".csv", ".json"}


def _normalize_markdown(content: str) -> str:
    """Collapse excessive blank lines and normalize whitespace.

    Reduces sequences of 3+ blank lines to 1, and strips trailing whitespace
    from each line (common in PDF conversions).
    """
    import re

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in content.splitlines(keepends=False)]
    # Collapse 3+ consecutive blank lines to 1
    text = "\n".join(lines)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text


_ORIGIN_COMMENT_PREFIX = "<!-- source:"


def _origin_comment(source_path: Path) -> str:
    """HTML-comment header recording the original file a Markdown file was converted from.

    Read back by `genai_graph.kg.query.markdown_tree_tui` to let the TUI offer opening the
    original document (PDF/DOCX/...) alongside the converted Markdown. Invisible when rendered.
    """
    return f"{_ORIGIN_COMMENT_PREFIX} {source_path.resolve()} -->\n\n"


class MarkdownizeManifestEntry(BaseModel):
    """A single markdown conversion entry in the manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarkdownizeManifest(BaseModel):
    """Manifest for markdown conversion results to avoid reprocessing."""

    entries: dict[str, MarkdownizeManifestEntry] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize manifest to JSON string."""
        return super().model_dump_json(**kwargs)


class MistralOCRBatchProcessor:
    """Submit PDFs to Mistral's OCR *batch* API and return Markdown text per file.

    Batching every PDF into a single job is markedly cheaper than per-file OCR
    calls, at the cost of polling until the job finishes.
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    async def process_batch(self, file_paths: list[Path]) -> dict[str, str]:
        """Return a mapping of ``str(pdf_path)`` to its extracted Markdown text."""
        import os

        from mistralai import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")

        client = Mistral(api_key=api_key)
        results: dict[str, str] = {}
        for start in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[start : start + self.batch_size]
            logger.info(f"Submitting Mistral OCR batch of {len(batch_files)} PDF(s)")
            requests = [self._prepare_batch_request(p, i) for i, p in enumerate(batch_files)]
            results.update(await self._submit_and_poll_batch(client, requests, batch_files))
        return results

    def _prepare_batch_request(self, file_path: Path, index: int) -> str:
        """Prepare a single batch request in JSONL format.

        Args:
            file_path: Path to file
            index: Index for custom_id

        Returns:
            JSONL formatted batch request line
        """
        import base64
        import json

        # Encode file to base64
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        document_url = f"data:application/pdf;base64,{content_b64}"

        request = {
            "custom_id": str(index),
            "body": {"model": "mistral-ocr-latest", "document": {"type": "document_url", "document_url": document_url}},
        }
        return json.dumps(request)

    async def _submit_and_poll_batch(
        self,
        client,
        batch_requests: list[str],
        file_paths: list[Path],
    ) -> dict[str, str]:
        """Submit one batch job, poll until done, and return per-PDF Markdown text."""
        import os
        import tempfile

        results: dict[str, str] = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(request + "\n")
            batch_file_path = f.name

        try:
            with open(batch_file_path, "rb") as f:
                batch_data = client.files.upload(
                    file={"file_name": os.path.basename(batch_file_path), "content": f},
                    purpose="batch",
                )

            job = client.batch.jobs.create(
                input_files=[batch_data.id],
                model="mistral-ocr-latest",
                endpoint="/v1/ocr",
                metadata={"job_type": "pdf_ocr_batch"},
            )

            logger.info(f"Polling Mistral batch job {job.id} for completion")
            if not await self._poll_job(client, job.id):
                raise RuntimeError("Mistral OCR batch job failed to complete")

            retrieved_job = client.batch.jobs.get(job_id=job.id)
            if retrieved_job.output_file:
                results = self._parse_results(client, retrieved_job.output_file, file_paths)

        finally:
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

        return results

    async def _poll_job(self, client, job_id: str, max_attempts: int = 300) -> bool:
        """Poll job status until completion.

        Args:
            client: Mistral client
            job_id: Batch job ID
            max_attempts: Maximum polling attempts

        Returns:
            True if job succeeded, False otherwise
        """

        for _attempt in range(max_attempts):
            job = client.batch.jobs.get(job_id=job_id)
            logger.info(f"Batch job status: {job.status}")

            if job.status == "SUCCESS":
                logger.success(f"Batch job {job_id} completed successfully")
                return True
            elif job.status == "FAILED":
                logger.error(f"Batch job {job_id} failed")
                return False

            # Wait before next poll
            await asyncio.sleep(2)

        logger.error(f"Batch job {job_id} did not complete within timeout")
        return False

    def _parse_results(self, client, output_file_id: str, file_paths: list[Path]) -> dict[str, str]:
        """Download the batch output and return ``str(pdf_path) -> Markdown text``."""
        import json

        from mistralai.models import OCRResponse

        results: dict[str, str] = {}

        logger.info("Downloading Mistral batch results")
        output_stream = client.files.download(file_id=output_file_id)
        response_content = output_stream.read().decode("utf-8")

        for line in response_content.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            file_path = file_paths[int(result["custom_id"])]
            response_body = result.get("response", {}).get("body", {})
            try:
                ocr_response = OCRResponse.model_validate(response_body)
                results[str(file_path)] = _ocr_response_to_markdown(ocr_response)
            except Exception as e:
                logger.error(f"Failed to parse OCR result for {file_path.name}: {e}")

        return results


def _ocr_response_to_markdown(ocr_response) -> str:
    """Join a Mistral OCR response's pages into a single Markdown string."""
    parts: list[str] = []
    for page in ocr_response.pages:
        parts.append(f"## Page {page.index + 1}\n\n{page.markdown}\n\n")
    return "".join(parts)


@dataclass(slots=True)
class _FileToProcess:
    path: Path
    content_hash: str


def _prepare_files(
    files: Iterable[Path],
    cache: ManifestCache,
    force: bool,
    already_processed: Callable[[str], bool] | None = None,
    code_version: str | None = None,
) -> tuple[list[_FileToProcess], int]:
    """Prepare files for processing, skipping unchanged entries in the cache.

    When ``already_processed`` is supplied it is consulted with each file's
    content hash first: returning True means a downstream store (e.g. a graph
    DB) already holds the converted output, so the file is skipped even if the
    local manifest cache is cold. This lets a caller inject a persistence-aware
    skip check without this module depending on that store.

    ``code_version`` (the markdownize profile fingerprint) is stored alongside
    the content fingerprint; a mismatch invalidates the cache even when the
    source file itself is unchanged, so switching profiles triggers reprocessing.
    """
    from genai_tk.utils.hashing import buffer_digest

    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = buffer_digest(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {path}: {exc}")
            continue

        if not force and already_processed is not None and already_processed(content_hash):
            skipped += 1
            logger.info(f"Skipping already-processed file (store hit): {path}")
            continue

        if cache.is_fresh(str(path), fingerprint=content_hash, force=force, code_version=code_version):
            skipped += 1
            logger.info(f"Skipping unchanged file: {path}")
            continue

        to_process.append(_FileToProcess(path=path, content_hash=content_hash))

    return to_process, skipped


def _is_markdownize_compatible(file_path: Path) -> bool:
    """Check if file is one of the supported source-document formats."""
    suffix = file_path.suffix.lower()
    return suffix in (PPT_EXTS | DOC_EXTS | EXCEL_EXTS | IMAGE_EXTS | DIRECT_MARKITDOWN_EXTS | {".pdf"})


def _route_for(file_path: Path, profile: MarkdownizeProfile) -> str:
    """Return the conversion route for a file under a profile.

    Routes: ``via_pdf`` (LibreOffice → PDF → pdf_converter), ``pdf`` (native PDF
    → pdf_converter), ``md_parser`` (spreadsheet parser), or ``markitdown``.
    """
    suffix = file_path.suffix.lower()
    if suffix in PPT_EXTS:
        return "via_pdf" if profile.ppt_converter == "via_pdf" else "markitdown"
    if suffix in DOC_EXTS:
        return "via_pdf" if profile.doc_converter == "via_pdf" else "markitdown"
    if suffix in EXCEL_EXTS:
        return profile.excel_converter  # via_pdf | markitdown | md_parser
    if suffix == ".pdf":
        return "pdf"
    return "markitdown"


def _grid_cell(value: Any) -> str:
    """Stringify a raw openpyxl cell value, mapping None to an empty string."""
    return "" if value is None else str(value)


def _drop_empty_rows_and_cols(grid: list[list[str]]) -> list[list[str]]:
    """Remove fully-blank rows/columns and pad ragged rows to a common width."""
    rows = [row for row in grid if any(cell.strip() for cell in row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep_cols = [i for i in range(width) if any(row[i].strip() for row in rows)]
    return [[row[i] for i in keep_cols] for row in rows]


def _split_leading_title(grid: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    """Rescue a title row (a single filled cell above a wider header row) from the header."""
    first_row = grid[0]
    filled = [cell for cell in first_row if cell.strip()]
    if len(grid) > 1 and len(filled) == 1 and sum(bool(cell.strip()) for cell in grid[1]) > 1:
        return filled[0], grid[1:]
    return None, grid


def _excel_to_markdown_md_parser(path: Path) -> str:
    """Convert an .xlsx/.xls file to Markdown via ``md-spreadsheet-parser``.

    One section per worksheet: empty rows/columns are dropped, a leading title
    row is promoted to a heading, and merged header cells are forward-filled.
    """
    import openpyxl
    from md_spreadsheet_parser import ExcelParsingSchema, parse_excel

    schema = ExcelParsingSchema(header_rows=1, fill_merged_headers=True)
    workbook = openpyxl.load_workbook(path, data_only=True)
    parts: list[str] = []

    for worksheet in workbook.worksheets:
        grid = [[_grid_cell(cell) for cell in row] for row in worksheet.iter_rows(values_only=True)]
        grid = _drop_empty_rows_and_cols(grid)
        if not grid:
            continue

        title, grid = _split_leading_title(grid)
        heading = f"## {worksheet.title}" + (f"\n\n### {title}" if title else "")
        table = parse_excel(grid, schema)
        parts.append(f"{heading}\n\n{table.to_markdown()}\n")

    return "\n".join(parts)


def _markitdown_text(path: Path) -> str:
    """Convert any markitdown-supported file to Markdown text."""
    from markitdown import MarkItDown

    return MarkItDown().convert(str(path)).text_content


def _edgeparse_text(path: Path) -> str | None:
    """Convert a PDF with edgeparse, returning None (to trigger fallback) on failure."""
    try:
        import edgeparse

        return edgeparse.convert(str(path), format="markdown")
    except Exception as e:
        logger.warning(f"edgeparse failed for {path.name}: {e}. Falling back to markitdown.")
        return None


def _convert_text(path: Path, route: str, pdf_converter: str) -> str:
    """Convert a single file to Markdown text for a non-Mistral route.

    ``route`` is one of ``md_parser`` / ``pdf`` / ``markitdown``. The ``pdf``
    route honours ``pdf_converter`` (``edgeparse`` with markitdown fallback, or
    ``markitdown``); Mistral PDFs are handled in the flow via the batch API.
    """
    if route == "md_parser":
        try:
            return _excel_to_markdown_md_parser(path)
        except Exception as e:
            logger.warning(f"md-spreadsheet-parser failed for {path.name}: {e}. Falling back to markitdown.")
            return _markitdown_text(path)
    if route == "pdf" and pdf_converter == "edgeparse":
        text = _edgeparse_text(path)
        if text is not None:
            return text
    return _markitdown_text(path)


def _output_paths(original: Path, root_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Return (relative, absolute) Markdown output paths preserving directory structure.

    The output filename embeds the source extension: ``review.xlsx`` → ``review_xlsx.md``.
    """
    try:
        rel = original.relative_to(root_dir)
    except ValueError:
        rel = Path(original.name)
    new_name = f"{rel.stem}_{rel.suffix.lstrip('.')}.md"
    rel_out = rel.parent / new_name
    return rel_out, output_dir / rel_out


def _write_markdown(output_file: Path, original: Path, content: str) -> None:
    """Write normalized Markdown with an origin comment pointing at the source file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(_origin_comment(original) + _normalize_markdown(content), encoding="utf-8")
    logger.success(f"Wrote markdown to {output_file}")


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


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _ocr_pdfs_with_mistral(
    pdf_items: list[tuple[_FileToProcess, str]],
    root_dir: str,
    output_dir: Path,
) -> list[tuple[str, str]]:
    """OCR every PDF in one Mistral batch job, writing Markdown at each original's path."""
    pdf_paths = [Path(pdf) for _, pdf in pdf_items]
    try:
        texts = asyncio.run(MistralOCRBatchProcessor().process_batch(pdf_paths))
    except Exception as e:
        logger.warning(f"Mistral batch OCR failed ({e}); falling back to markitdown for PDFs.")
        texts = {}

    results: list[tuple[str, str]] = []
    for file_info, pdf in pdf_items:
        rel_out, out_abs = _output_paths(file_info.path, Path(root_dir), output_dir)
        text = texts.get(str(pdf))
        if text is None:
            logger.info(f"markitdown fallback for {Path(pdf).name}")
            text = _markitdown_text(Path(pdf))
        _write_markdown(out_abs, file_info.path, text)
        results.append((str(file_info.path), str(rel_out)))
    return results


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
    base_dir: str,
    output_dir: str,
    *,
    profile: str | MarkdownizeProfile = "default",
    pathspecs: list[str] | None = None,
    batch_size: int = 5,
    force: bool = False,
    already_processed: Callable[[str], bool] | None = None,
) -> MarkdownizeManifest:
    """Convert a directory of documents to Markdown using a markdownize profile.

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        output_dir: Directory to write Markdown files and the manifest.
        profile: A :class:`MarkdownizeProfile` or the name of one (``fast`` /
            ``medium`` / ``best`` / ``default``, or a key configured under
            ``markdownize_profiles``).  The profile decides, per source-document
            family, whether to convert directly (markitdown / md_parser) or
            ``via_pdf`` (LibreOffice → PDF), and which ``pdf_converter`` turns
            every PDF into Markdown.  When ``pdf_converter`` is ``mistral`` all
            PDFs are sent in a single, cheaper Mistral batch job.
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).  Defaults to
            all supported document extensions.
        batch_size: Number of files converted concurrently per batch.
        force: Reprocess files even if unchanged in the manifest.
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
            for ext in sorted(PPT_EXTS | DOC_EXTS | EXCEL_EXTS | IMAGE_EXTS | DIRECT_MARKITDOWN_EXTS | {".pdf"})
        ]

    file_paths = resolve_files(base_dir, pathspecs=pathspecs)

    if not file_paths:
        logger.warning("No files found to process")
        return MarkdownizeManifest()

    logger.info(f"Discovered {len(file_paths)} files to process")

    resolved_base = resolve_config_path(base_dir)
    resolved_output = resolve_config_path(output_dir)
    output_upath = Path(resolved_output)
    output_upath.mkdir(parents=True, exist_ok=True)

    manifest_path = output_upath / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    code_version = resolved_profile.fingerprint()
    files = [Path(p) for p in file_paths if _is_markdownize_compatible(Path(p))]
    to_process, skipped = _prepare_files(
        files, cache, force=force, already_processed=already_processed, code_version=code_version
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
        pdf_root = output_upath / "_via_pdf"
        logger.info(f"Converting {len(via_pdf_files)} document(s) to PDF via LibreOffice")
        for batch in _chunked(via_pdf_files, batch_size):
            futures = [_libreoffice_task.submit(str(fi.path), str(pdf_root), resolved_base) for fi in batch]
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
            results.extend(_ocr_pdfs_with_mistral(pdf_items, resolved_base, output_upath))
        else:
            for batch in _chunked(pdf_items, batch_size):
                futures = []
                for fi, pdf in batch:
                    rel_out, out_abs = _output_paths(fi.path, Path(resolved_base), output_upath)
                    futures.append(
                        _convert_file_task.submit(
                            str(fi.path), pdf, "pdf", str(out_abs), str(rel_out), resolved_profile.pdf_converter
                        )
                    )
                results.extend(f.result() for f in futures)  # type: ignore[misc]

    # 4. Direct conversions: md_parser spreadsheets + markitdown for everything else.
    direct_files = [fi for fi in to_process if routes[str(fi.path)] in ("md_parser", "markitdown")]
    for batch in _chunked(direct_files, batch_size):
        futures = []
        for fi in batch:
            rel_out, out_abs = _output_paths(fi.path, Path(resolved_base), output_upath)
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
