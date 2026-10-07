"""Prefect flow cloning and ingesting Git repositories into Markdown files.

One task per repository: clones or updates the repo in cache, then dispatches
its files through ``markdownize_flow`` to convert documents (Markdown, Jupyter
notebooks, PDFs, Word docs) into normalized Markdown under ``md_output_dir``.

Routing note: this flow is the target of ``git_repo`` ingest rules (e.g.
``https://github.com/**``, ``https://gitlab.com/**``, ``**/*.git``, ``git@**``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner

from genai_tk.config_mgmt.file_patterns import resolve_config_path
from genai_tk.workflow.force import ForceStage, stage_active
from genai_tk.workflow.markdownize.flow import markdownize_flow
from genai_tk.workflow.markdownize.routing import ALL_DOCUMENT_EXTS
from genai_tk.workflow.registry import workflow
from genai_tk.workflow.sources import clone_git_repo, git_repo_slug, is_git_url, parse_git_url


def _get_commit_sha(repo_dir: Path) -> str:
    """Return HEAD commit SHA for a cloned git repository."""
    try:
        git_dir = repo_dir if (repo_dir / ".git").is_dir() else repo_dir.parent
        res = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


@task(log_prints=False, retries=2, retry_delay_seconds=5)
def _clone_repo_task(
    repo_url: str,
    cache_root: str,
    ref: str | None = None,
    subpath: str | None = None,
    force: bool = False,
) -> tuple[str, str, str, str]:
    """Clone or update one git repository. Returns (repo_url, slug, target_dir, sha)."""
    clean_url, url_ref, url_subpath = parse_git_url(repo_url)
    effective_ref = ref or url_ref
    effective_subpath = subpath or url_subpath
    slug = git_repo_slug(clean_url)
    local_path = clone_git_repo(
        clean_url,
        Path(cache_root),
        ref=effective_ref,
        subpath=effective_subpath,
        force=force,
    )
    sha = _get_commit_sha(local_path)
    return repo_url, slug, str(local_path), sha


@flow(name="git_repo", task_runner=ThreadPoolTaskRunner(max_workers=8))  # type: ignore[call-overload]
def git_repo_flow(
    sources: list[str],
    md_output_dir: str,
    *,
    ref: str | None = None,
    subpath: str | None = None,
    pathspecs: list[str] | None = None,
    profile: str = "default",
    force_stage: str | ForceStage | None = None,
) -> dict[str, Any]:
    """Clone git repositories and convert their documents to Markdown under *md_output_dir*.

    Args:
        sources: Git repository URLs (https, git@, ssh, or *.git).
        md_output_dir: Directory to write the resulting Markdown files.
        ref: Git branch, tag, or commit to checkout.
        subpath: Subdirectory inside the repository to ingest (e.g. 'docs').
        pathspecs: Gitwildmatch file patterns to include; defaults to document files excluding .git.
        profile: Markdownize profile for document conversion (e.g. 'default', 'docling').
        force_stage: Stage forcing re-execution; 'md' or 'all' forces re-cloning/pulling and re-conversion.

    Returns:
        Summary dict with ``processed``, ``skipped``, ``failed`` counts, ``repos`` metadata, and warnings.
    """
    git_sources = [
        s
        for s in sources
        if is_git_url(s) or s.startswith(("http://", "https://", "git@", "ssh://", "git://", "file://"))
    ]
    if not git_sources:
        logger.warning("git_repo_flow: no Git repository sources to process")
        return {"processed": 0, "skipped": 0, "failed": 0, "warnings": ["no Git repository sources"]}

    output_root = Path(resolve_config_path(md_output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    force = stage_active(force_stage, ForceStage.md) or stage_active(force_stage, ForceStage.all)

    effective_pathspecs = (
        list(pathspecs) if pathspecs is not None else [f"**/*{ext}" for ext in sorted(ALL_DOCUMENT_EXTS)]
    )
    effective_pathspecs.extend(["!**/.git/**", "!**/.git"])

    processed_total = 0
    skipped_total = 0
    failed_total = 0
    warnings: list[str] = []
    repos_info: list[dict[str, Any]] = []

    futures = [
        _clone_repo_task.submit(
            src,
            str(cache_root),
            ref=ref,
            subpath=subpath,
            force=force,
        )
        for src in git_sources
    ]

    for src, future in zip(git_sources, futures, strict=True):
        try:
            repo_url, slug, local_dir, sha = future.result()
            if output_root.name == slug or len(git_sources) == 1:
                dest_dir = output_root
            else:
                dest_dir = output_root / slug
            dest_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Markdownizing git repo {} -> {}", slug, dest_dir)
            manifest = markdownize_flow(
                sources=[local_dir],
                md_output_dir=str(dest_dir),
                cache_dir=str(cache_root / "manifests" / slug),
                profile=profile,
                pathspecs=effective_pathspecs,
                force_stage=force_stage,
            )
            count = len(manifest.entries)
            processed_total += count
            repos_info.append(
                {
                    "url": repo_url,
                    "slug": slug,
                    "sha": sha,
                    "files_count": count,
                    "output_dir": str(dest_dir),
                }
            )
            logger.success("Git repo {} ingested: {} document(s) -> {}", slug, count, dest_dir)
        except Exception as exc:
            logger.error("Failed to ingest git repo {}: {}", src, exc)
            failed_total += 1
            warnings.append(f"Git repo ingest failed for {src}: {exc}")

    return {
        "processed": processed_total,
        "skipped": skipped_total,
        "failed": failed_total,
        "md_output_dir": str(output_root),
        "repos": repos_info,
        "warnings": warnings,
    }


@workflow(name="git_repo", description="Clone and ingest git repository to Markdown")
def git_repo_step(
    *,
    sources: list[str],
    md_output_dir: str,
    ref: str | None = None,
    subpath: str | None = None,
    pathspecs: list[str] | None = None,
    profile: str = "default",
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Workflow-engine wrapper around :func:`git_repo_flow`."""
    return git_repo_flow(
        sources=sources,
        md_output_dir=md_output_dir,
        ref=ref,
        subpath=subpath,
        pathspecs=pathspecs,
        profile=profile,
        force_stage=force_stage,
    )
