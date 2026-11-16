"""Shared skill-source resolution for every agent harness.

Skill directories listed in an agent profile may reference skills shipped
inside an installed library using a ``package:<distribution>/<subpath>``
scheme (e.g. ``package:genai_graph/agent/skills``). Resolution happens at the
harness-agnostic profile layer (see
:mod:`genai_tk.agents.harness.profiles`) so every harness — langchain,
deerflow, docgraph — receives plain filesystem paths; individual harnesses
need no knowledge of the scheme.
"""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

from loguru import logger

_PACKAGE_SCHEME = "package:"


def resolve_package_dir(entry: str) -> str | None:
    """Resolve a ``package:<distribution>/<subpath>`` entry to a filesystem path.

    Editable checkouts resolve to the live source, wheels to site-packages.
    Returns None when the entry is not a ``package:`` reference or the package
    or path does not exist.
    """
    if not entry.startswith(_PACKAGE_SCHEME):
        return None
    parts = entry[len(_PACKAGE_SCHEME) :].strip("/").split("/")
    if not parts or not parts[0]:
        logger.warning("Invalid package skill reference '{}' (expected package:<name>/<path>).", entry)
        return None
    try:
        traversable = pkg_files(parts[0])
        for part in parts[1:]:
            traversable = traversable / part
        path = Path(str(traversable))
    except Exception as exc:
        logger.warning("Package skill reference '{}' could not be resolved: {}", entry, exc)
        return None
    if not path.is_dir():
        logger.warning("Package skill reference '{}' does not point at an existing directory.", entry)
        return None
    return str(path)


def resolve_skill_sources(entries: list[str]) -> list[str]:
    """Translate ``package:`` references to real paths; keep plain paths as-is.

    Unresolvable ``package:`` references are dropped with a warning so an
    optional dependency never breaks agent startup. Duplicates are removed
    while preserving order.
    """
    resolved: list[str] = []
    for entry in entries:
        package_dir = resolve_package_dir(entry)
        if package_dir:
            resolved.append(package_dir)
        elif not entry.startswith(_PACKAGE_SCHEME):
            resolved.append(entry)
    return list(dict.fromkeys(resolved))
