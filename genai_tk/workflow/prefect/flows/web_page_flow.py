"""Prefect flow fetching web pages (URLs) into Markdown files.

One task per URL: failures are isolated, retries are cheap (manifest cache),
and results converge to the same artifact contract as ``markdownize_flow`` —
Markdown under ``md_output_dir`` plus a manifest entry keyed by the source URL.

Routing note: this flow is the target of ``web_page`` ingest rules; URLs are
fetched once and re-fetched only with ``force_stage`` or when the fetcher
changes (the fetcher name is part of the cache code-version).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import BaseModel

from genai_tk.utils.hashing import buffer_digest
from genai_tk.web.fetchers import get_web_page_fetcher
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.force import ForceStage, stage_active
from genai_tk.workflow.registry import workflow


class _FetchResult(BaseModel):
    url: str
    output_path: str | None = None
    error: str | None = None


def _url_slug(url: str) -> str:
    """Return a filesystem-safe, collision-free name for a URL."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", url.removesuffix("https://").removesuffix("http://")).strip("-")
    slug = slug[:80] or "page"
    digest = buffer_digest(url.encode())[:8]
    return f"{slug}_{digest}.md"


@task(log_prints=False, retries=2, retry_delay_seconds=5)
def _fetch_url_task(url: str, fetcher_name: str) -> str:
    """Fetch one URL and return its Markdown content."""
    fetcher = get_web_page_fetcher(fetcher_name)
    return fetcher.fetch(url)


@flow(name="web_page", task_runner=ThreadPoolTaskRunner(max_workers=8))  # type: ignore[call-overload]
def web_page_flow(
    sources: list[str],
    md_output_dir: str,
    *,
    fetcher: str = "bs",
    force_stage: str | ForceStage | None = None,
) -> dict[str, Any]:
    """Fetch URLs and write each page as Markdown under *md_output_dir*.

    Args:
        sources: http(s) URLs to fetch; non-URL items are ignored.
        md_output_dir: Directory to write the Markdown files.
        fetcher: Fetcher implementation name — 'bs' (BeautifulSoup, local) or
            'tavily' (Tavily Extract API).
        force_stage: Stage forcing re-execution; 'md' or 'all' re-fetch cached URLs.

    Returns:
        Summary dict with ``processed``, ``skipped``, ``failed`` counts and warnings.
    """
    urls = [s for s in sources if s.startswith("http://") or s.startswith("https://")]
    if not urls:
        logger.warning("web_page_flow: no URL sources to process")
        return {"processed": 0, "skipped": 0, "failed": 0, "warnings": ["no URL sources"]}

    output_dir = _mkdirs(md_output_dir)
    cache = ManifestCache.load(output_dir / ".cache" / "manifest.json")
    code_version = f"web_page:{fetcher}"
    force = stage_active(force_stage, ForceStage.md) or stage_active(force_stage, ForceStage.all)

    to_fetch: list[str] = []
    skipped = 0
    for url in urls:
        if cache.is_fresh(url, fingerprint=buffer_digest(url.encode()), force=force, code_version=code_version):
            skipped += 1
        else:
            to_fetch.append(url)

    results: list[_FetchResult] = []
    for chunk in _chunks(to_fetch, 5):
        futures = [_fetch_url_task.submit(url, fetcher) for url in chunk]
        for url, future in zip(chunk, futures, strict=True):
            try:
                markdown = future.result()
                out_rel = _url_slug(url)
                out_abs = output_dir / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                out_abs.write_text(_front_matter(url) + markdown, encoding="utf-8")
                cache.record_success(
                    key=url,
                    fingerprint=buffer_digest(url.encode()),
                    code_version=code_version,
                    outputs={"output_path": out_rel, "fetched_at": _now(), "content_hash": buffer_digest(markdown.encode())},
                )
                results.append(_FetchResult(url=url, output_path=out_rel))
                logger.success("Fetched {} -> {}", url, out_abs)
            except Exception as exc:
                logger.error("Failed to fetch {}: {}", url, exc)
                results.append(_FetchResult(url=url, error=str(exc)))

    (output_dir / ".cache").mkdir(parents=True, exist_ok=True)
    cache.save(output_dir / ".cache" / "manifest.json")

    failed = [r for r in results if r.error]
    return {
        "processed": len(results) - len(failed),
        "skipped": skipped,
        "failed": len(failed),
        "md_output_dir": str(output_dir),
        "warnings": [f"web_page fetch failed for {r.url}: {r.error}" for r in failed],
    }


def _mkdirs(md_output_dir: str):
    from pathlib import Path

    from genai_tk.config_mgmt.file_patterns import resolve_config_path

    path = Path(resolve_config_path(md_output_dir))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)] if size > 0 else [items]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _front_matter(url: str) -> str:
    return f"---\nsource_url: {url}\nfetched_at: {_now()}\n---\n\n"


@workflow(name="web_page", description="Fetch web pages (URLs) to Markdown via the configured fetcher")
def web_page_step(
    *,
    sources: list[str],
    md_output_dir: str,
    fetcher: str = "bs",
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Workflow-engine wrapper around :func:`web_page_flow`."""
    return web_page_flow(sources=sources, md_output_dir=md_output_dir, fetcher=fetcher, force_stage=force_stage)
