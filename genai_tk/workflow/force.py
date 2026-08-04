"""Unified, staged cache-invalidation model for document/KG pipelines.

A single ordered ``--force <stage>`` replaces the old collection of ad-hoc
booleans (``force``, ``force_rebuild``, ``--remarkdownize``). Forcing a stage
re-runs it *and every stage downstream of it*, since downstream caches are
derived from upstream outputs.

Stage order (index, low → high)::

    unzip < pdf < md < parquet < graph < embed < all

- ``unzip``   — re-extract zip archives (markdownize, doctree)
- ``pdf``     — re-run office→PDF conversion (markdownize, doctree)
- ``md``      — re-run document→Markdown conversion (markdownize, doctree)
- ``parquet`` — re-export JSON→parquet import caches (kg build)
- ``graph``   — re-ingest into the graph database (doctree, kg build)
- ``embed``   — recompute embeddings
- ``all``     — force every stage, including dropping the destination store

Each command exposes only the stages relevant to it (e.g. ``markdownize``
only cares about ``unzip``/``pdf``/``md``) via :func:`stage_active`.
"""

from __future__ import annotations

from enum import Enum


class ForceStage(str, Enum):
    """An ordered cache-invalidation checkpoint in a document/KG pipeline."""

    unzip = "unzip"
    pdf = "pdf"
    md = "md"
    parquet = "parquet"
    graph = "graph"
    embed = "embed"
    all = "all"


_ORDER: list[ForceStage] = [
    ForceStage.unzip,
    ForceStage.pdf,
    ForceStage.md,
    ForceStage.parquet,
    ForceStage.graph,
    ForceStage.embed,
    ForceStage.all,
]


def stage_active(requested: str | ForceStage | None, stage: str | ForceStage) -> bool:
    """Return True when *stage* must be force-invalidated given *requested*.

    Forcing a stage implies every downstream stage is also forced (e.g.
    requesting ``md`` also forces ``parquet``, ``graph``, and ``embed``).
    ``all`` always forces every stage. ``None`` never forces anything.

    Example:
        ```python
        stage_active("md", "graph")  # True — md forces downstream graph rebuild
        stage_active("graph", "md")  # False — graph doesn't force upstream md
        stage_active(None, "md")  # False
        stage_active("all", "unzip")  # True
        ```
    """
    if requested is None:
        return False
    requested = ForceStage(requested)
    if requested is ForceStage.all:
        return True
    stage = ForceStage(stage)
    return _ORDER.index(requested) <= _ORDER.index(stage)
