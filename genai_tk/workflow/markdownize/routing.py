"""File discovery and output path helpers for markdownize flow."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from genai_tk.extra.markdownize.selector import MarkdownizeProfile
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.markdownize.text import _normalize_markdown, _origin_comment
from genai_tk.workflow.sources import ResolvedSourceFile

PPT_EXTS = {".ppt", ".pptx", ".odp", ".pps", ".pot", ".pptm", ".ppsx", ".ppsm"}
DOC_EXTS = {".doc", ".docx", ".odt", ".rtf", ".docm"}
EXCEL_EXTS = {".xls", ".xlsx", ".ods", ".xlsm", ".xlsb"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".gif", ".bmp", ".webp"}
DIRECT_MARKITDOWN_EXTS = {".html", ".htm", ".csv", ".json"}
MD_EXTS = {".md", ".markdown"}

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
    """Prepare files for processing, skipping unchanged entries in the cache."""
    from genai_tk.utils.hashing import buffer_digest

    to_process: list[_FileToProcess] = []
    skipped = 0

    for rf in files:
        try:
            content_bytes = rf.path.read_bytes()
            content_hash = buffer_digest(content_bytes)
        except Exception as exc:
            logger.error(f"Error reading {rf.path}: {exc}")
            continue

        if not force and already_processed is not None and already_processed(content_hash):
            skipped += 1
            continue

        if cache.is_fresh(str(rf.path), fingerprint=content_hash, force=force, code_version=code_version):
            skipped += 1
            continue

        to_process.append(_FileToProcess(path=rf.path, content_hash=content_hash, root=rf.root))

    return to_process, skipped


def _is_markdownize_compatible(file_path: Path) -> bool:
    """Check if file is one of the supported source-document formats."""
    return file_path.suffix.lower() in ALL_DOCUMENT_EXTS


def _route_for(file_path: Path, profile: MarkdownizeProfile) -> str:
    """Return the converter or route name for a file under a profile."""
    return profile.select_route(file_path)


def _output_paths(original: Path, root_dir: Path, output_dir: Path, *, route: str = "convert") -> tuple[Path, Path]:
    """Return (relative, absolute) Markdown output paths preserving directory structure."""
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
