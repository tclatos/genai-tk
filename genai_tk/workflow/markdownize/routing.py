"""File discovery, routing, and output-path logic for markdownize_flow."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from genai_tk.extra.markdownize.selector import MarkdownizeProfile
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.markdownize.text import _normalize_markdown, _origin_comment
from genai_tk.workflow.sources import ResolvedSourceFile

# Source-document families and how markdownize_flow routes each suffix.
PPT_EXTS = {".ppt", ".pptx", ".odp"}
DOC_EXTS = {".doc", ".docx", ".odt", ".rtf"}
EXCEL_EXTS = {".xls", ".xlsx", ".ods"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".gif", ".bmp"}
DIRECT_MARKITDOWN_EXTS = {".html", ".htm", ".csv", ".json"}
# Pre-existing Markdown: copied through unchanged rather than converted.
MD_EXTS = {".md", ".markdown"}
MD_EXTENSIONS = MD_EXTS

ALL_DOCUMENT_EXTS = PPT_EXTS | DOC_EXTS | EXCEL_EXTS | IMAGE_EXTS | DIRECT_MARKITDOWN_EXTS | MD_EXTS | {".pdf"}


@dataclass(slots=True)
class _FileToProcess:
    path: Path
    content_hash: str
    root: Path


def _prepare_files(
    files: Iterable[ResolvedSourceFile],
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

    for rf in files:
        try:
            content_bytes = rf.path.read_bytes()
            content_hash = buffer_digest(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {rf.path}: {exc}")
            continue

        if not force and already_processed is not None and already_processed(content_hash):
            skipped += 1
            logger.info(f"Skipping already-processed file (store hit): {rf.path}")
            continue

        if cache.is_fresh(str(rf.path), fingerprint=content_hash, force=force, code_version=code_version):
            skipped += 1
            logger.info(f"Skipping unchanged file: {rf.path}")
            continue

        to_process.append(_FileToProcess(path=rf.path, content_hash=content_hash, root=rf.root))

    return to_process, skipped


def _is_markdownize_compatible(file_path: Path) -> bool:
    """Check if file is one of the supported source-document formats."""
    suffix = file_path.suffix.lower()
    return suffix in ALL_DOCUMENT_EXTS


def _route_for(file_path: Path, profile: MarkdownizeProfile) -> str:
    """Return the conversion route or converter name for a file under a profile.

    Routes: ``copy`` (pre-existing Markdown, passed through unchanged),
    ``via_pdf`` (LibreOffice → PDF → pdf_converter), ``pdf`` (native PDF
    → pdf_converter), or specific converter names (e.g. ``messy_xls``,
    ``markitdown``, ``lighton_ocr``, ``anydoc``, ``llm``).
    """
    suffix = file_path.suffix.lower()
    if suffix in MD_EXTENSIONS:
        return "copy"

    # Use pathspec-based profile rules
    route = profile.select_route(file_path)
    if route == "messy_xls_parser":
        return "messy_xls_parser"
    if route in ("mistral", "mistral_ocr") and suffix == ".pdf":
        return "pdf"
    if suffix == ".pdf" and route != "via_pdf":
        return "pdf"

    return route


def _output_paths(original: Path, root_dir: Path, output_dir: Path, *, route: str = "convert") -> tuple[Path, Path]:
    """Return (relative, absolute) Markdown output paths preserving directory structure.

    The output filename embeds the source extension: ``review.xlsx`` → ``review_xlsx.md``.
    The ``copy`` route (pre-existing Markdown) keeps the original filename as-is.
    """
    try:
        rel = original.relative_to(root_dir)
    except ValueError:
        rel = Path(original.name)
    if route == "copy":
        return rel, output_dir / rel
    new_name = f"{rel.stem}_{rel.suffix.lstrip('.')}.md"
    rel_out = rel.parent / new_name
    return rel_out, output_dir / rel_out


def _write_markdown(output_file: Path, original: Path, content: str) -> None:
    """Write normalized Markdown with an origin comment pointing at the source file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(_origin_comment(original) + _normalize_markdown(content), encoding="utf-8")
    logger.success(f"Wrote markdown to {output_file}")
