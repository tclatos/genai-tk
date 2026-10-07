"""Resolve directory / ``.zip`` / file input specs to a flat list of files.

Shared by document-processing flows (``markdownize_flow``, ``office2pdf_flow``)
and downstream graph-ingestion commands (e.g. genai-graph's ``doctree build``)
so zip extraction and file discovery live in exactly one place.
"""

from __future__ import annotations

import re
import shutil
import subprocess
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


def is_git_url(item: str) -> bool:
    """Return True when *item* is a git repository URL or spec."""
    if not isinstance(item, str):
        return False
    clean = item.strip()
    if clean.startswith(("git@", "git://", "ssh://", "file://")):
        return True
    if clean.endswith((".git", ".git/")):
        return True
    m = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$", clean)
    if m:
        host = m.group(1).lower()
        if host in ("github.com", "gitlab.com", "bitbucket.org", "dev.azure.com") or host.startswith(
            ("git.", "gitlab.", "github.")
        ):
            return True
    return False


def parse_git_url(item: str) -> tuple[str, str | None, str | None]:
    """Parse a git URL into (clean_repo_url, ref, subpath).

    Examples:
        >>> parse_git_url("https://github.com/owner/repo")
        ("https://github.com/owner/repo", None, None)
        >>> parse_git_url("https://github.com/owner/repo/tree/main/docs")
        ("https://github.com/owner/repo", "main", "docs")
    """
    clean = item.strip().rstrip("/")
    m = re.match(r"^(https?://[^/]+/[^/]+/[^/]+)/(?:tree|blob)/([^/]+)(?:/(.+))?$", clean)
    if m:
        base_repo = m.group(1)
        ref = m.group(2)
        subpath = m.group(3)
        return base_repo, ref, subpath
    return clean, None, None


def git_repo_slug(repo_url: str) -> str:
    """Extract a clean repository name/slug from a git URL."""
    clean = repo_url.strip().rstrip("/")
    clean = clean.removesuffix(".git")
    name = Path(clean).name
    if ":" in name and not name.startswith("http"):
        name = name.split(":")[-1]
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name or "repo"


def clone_git_repo(
    repo_url: str,
    cache_dir: Path,
    *,
    ref: str | None = None,
    subpath: str | None = None,
    force: bool = False,
) -> Path:
    """Shallow-clone or update a git repo in cache_dir, returning the directory to process.

    Args:
        repo_url: Git repository URL (https, git@, ssh, etc.).
        cache_dir: Root cache directory (git clones are placed in ``cache_dir/git/<slug>_<digest>``).
        ref: Git branch, tag, or commit to clone or checkout.
        subpath: Optional subdirectory inside the repository.
        force: Re-fetch / reset even if already cloned.
    """
    from genai_tk.utils.hashing import buffer_digest

    clean_url, url_ref, url_subpath = parse_git_url(repo_url)
    effective_ref = ref or url_ref
    effective_subpath = subpath or url_subpath

    slug = git_repo_slug(clean_url)
    digest = buffer_digest(f"{clean_url}@{effective_ref or 'default'}".encode("utf-8"))[:8]
    target_repo_dir = cache_dir / "git" / f"{slug}_{digest}"

    if target_repo_dir.exists() and (target_repo_dir / ".git").is_dir():
        if force:
            logger.info("Force-updating git repo {} in {}", clean_url, target_repo_dir)
            try:
                cmd = ["git", "-C", str(target_repo_dir), "fetch", "--depth", "1", "origin"]
                if effective_ref:
                    cmd.append(effective_ref)
                subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                reset_cmd = ["git", "-C", str(target_repo_dir), "reset", "--hard", "FETCH_HEAD"]
                subprocess.run(reset_cmd, capture_output=True, text=True, timeout=30, check=True)
            except Exception as exc:
                logger.warning("Failed to force-update git repo {}, re-cloning: {}", clean_url, exc)
                shutil.rmtree(target_repo_dir)
                _do_clone(clean_url, target_repo_dir, effective_ref)
        else:
            try:
                cmd = ["git", "-C", str(target_repo_dir), "pull", "--ff-only"]
                subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except Exception as exc:
                logger.debug("git pull failed for {} (using cached clone): {}", clean_url, exc)
    else:
        if target_repo_dir.exists():
            shutil.rmtree(target_repo_dir)
        target_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _do_clone(clean_url, target_repo_dir, effective_ref)

    if effective_subpath:
        sub_dir = target_repo_dir / effective_subpath
        if not sub_dir.exists():
            logger.warning("Subpath '{}' not found in repo {}, using root", effective_subpath, clean_url)
            return target_repo_dir
        return sub_dir

    return target_repo_dir


def _do_clone(clean_url: str, dest_dir: Path, ref: str | None) -> None:
    logger.info("Cloning {} -> {}", clean_url, dest_dir)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [clean_url, str(dest_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"git clone failed for {clean_url}: {res.stderr.strip()}")


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
    - A git repository URL is cloned into ``cache_dir/git/`` and then walked.
    - A single file is returned as-is (its parent directory becomes its root).

    Args:
        sources: One source, or a list of sources — directories, ``.zip``
            archives, git URLs, or individual files. Supports ``${paths.*}`` config vars.
        cache_dir: Root cache directory for zip extraction and git clones.
        pathspecs: Gitwildmatch patterns applied when walking directories.
        force_unzip: Re-extract zip archives / re-fetch git repos even if already cached.

    Example:
        ```python
        files = resolve_sources(["./docs", "./archive.zip"], cache_dir=Path("./out/.cache"))
        ```
    """
    specs = [sources] if isinstance(sources, str) else list(sources)
    resolved: list[ResolvedSourceFile] = []
    for spec in specs:
        if is_git_url(spec):
            root = clone_git_repo(spec, cache_dir, force=force_unzip)
            effective_pathspecs = list(pathspecs) if pathspecs is not None else ["**/*"]
            effective_pathspecs = list(effective_pathspecs) + ["!**/.git/**", "!**/.git"]
            resolved.extend(
                ResolvedSourceFile(path=Path(f), root=root)
                for f in resolve_files(str(root), pathspecs=effective_pathspecs)
            )
            continue
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
