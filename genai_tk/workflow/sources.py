"""Resolve directory / ``.zip`` / file input specs to a flat list of files.

Shared by document-processing flows (``markdownize_flow``, ``office2pdf_flow``)
and downstream graph-ingestion commands (e.g. genai-graph's ``doctree build``)
so zip extraction and file discovery live in exactly one place.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from genai_tk.config_mgmt.file_patterns import resolve_config_path, resolve_files


class ResolvedSourceFile(BaseModel):
    """A discovered file plus the root directory its relative path is computed against."""

    path: Path
    root: Path

    model_config = {"arbitrary_types_allowed": True}

    @property
    def relative_path(self) -> Path:
        """Path of ``path`` relative to ``root`` (falls back to just the filename)."""
        try:
            return self.path.relative_to(self.root)
        except ValueError:
            return Path(self.path.name)


def extract_zip(zip_path: Path, cache_dir: Path, *, force: bool = False) -> Path:
    """Extract *zip_path* into ``cache_dir/unzipped/<stem>_<digest>`` (idempotent).

    Args:
        zip_path: Path to the ``.zip`` archive.
        cache_dir: Root cache directory (the ``unzipped/`` subfolder is created under it).
        force: Re-extract even if the target directory already exists.
    """
    from genai_tk.utils.hashing import buffer_digest

    digest = buffer_digest(str(zip_path.resolve()).encode("utf-8"))
    extract_dir = cache_dir / "unzipped" / f"{zip_path.stem}_{digest}"
    if extract_dir.exists() and not force:
        return extract_dir

    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path.name} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def resolve_sources(
    sources: str | list[str],
    *,
    cache_dir: Path,
    pathspecs: list[str] | None = None,
    force_unzip: bool = False,
) -> list[ResolvedSourceFile]:
    """Resolve directory / ``.zip`` / file specs into a flat list of files with roots.

    - A directory is walked with *pathspecs* (gitwildmatch, ``!`` = exclude).
    - A ``.zip`` archive is extracted into ``cache_dir/unzipped/`` and then walked.
    - A single file is returned as-is (its parent directory becomes its root).

    Args:
        sources: One source, or a list of sources — directories, ``.zip``
            archives, or individual files. Supports ``${paths.*}`` config vars.
        cache_dir: Root cache directory for zip extraction.
        pathspecs: Gitwildmatch patterns applied when walking directories.
        force_unzip: Re-extract zip archives even if already cached.

    Example:
        ```python
        files = resolve_sources(["./docs", "./archive.zip"], cache_dir=Path("./out/.cache"))
        ```
    """
    specs = [sources] if isinstance(sources, str) else list(sources)
    resolved: list[ResolvedSourceFile] = []
    for spec in specs:
        source_path = Path(resolve_config_path(spec))
        if source_path.is_dir():
            root = source_path
        elif source_path.suffix.lower() == ".zip":
            root = extract_zip(source_path, cache_dir, force=force_unzip)
        elif source_path.is_file():
            resolved.append(ResolvedSourceFile(path=source_path, root=source_path.parent))
            continue
        else:
            logger.warning(f"Source not found: {source_path}")
            continue
        resolved.extend(
            ResolvedSourceFile(path=Path(f), root=root) for f in resolve_files(str(root), pathspecs=pathspecs)
        )
    return resolved
