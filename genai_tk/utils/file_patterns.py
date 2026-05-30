"""File matching utilities using .gitignore-style pathspecs.

Provides two thin functions:

- ``resolve_config_path`` -- expand ``${config.key}`` references in a path string.
- ``resolve_files``       -- walk *base_dir* and return files matched by pathspecs.
  Patterns starting with ``!`` exclude files, exactly as in ``.gitignore``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pathspec
from loguru import logger


def resolve_config_path(path_str: str) -> str:
    """Expand ``${config.key}`` references in a path string using the app config.

    Args:
        path_str: Path string that may contain ``${section.key}`` references.

    Returns:
        Resolved path string; unresolvable references are kept as-is.
    """
    from genai_tk.utils.config_mngr import global_config

    _pat = re.compile(r"\$\{([^}]+)\}")

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        try:
            val = global_config().get_str(key)
            if val is None:
                logger.warning("Config key {!r} not found", key)
                return m.group(0)
            return str(val)
        except Exception as exc:
            logger.warning("Failed to resolve config key {!r}: {}", key, exc)
            return m.group(0)

    return _pat.sub(_sub, path_str)


def resolve_files(
    base_dir: str,
    *,
    pathspecs: list[str] | None = None,
) -> list[Path]:
    """Return all files under *base_dir* matched by *pathspecs*.

    Pathspecs follow ``.gitignore`` / gitwildmatch semantics: plain patterns
    select files; patterns prefixed with ``!`` de-select previously selected
    files.  ``${config.key}`` references in *base_dir* are expanded.

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        pathspecs: List of gitwildmatch patterns.  Defaults to ``["**/*"]``
            (all files, recursive).

    Returns:
        Sorted list of matched Path objects.

    Example:
        ```python
        files = resolve_files(
            base_dir="${paths.data_root}/docs",
            pathspecs=["**/*.pdf", "!**/*_draft*"],
        )
        ```
    """
    resolved_root = resolve_config_path(base_dir)
    root = Path(resolved_root)

    if not root.exists():
        logger.error("base_dir does not exist: {}", root)
        return []
    if not root.is_dir():
        logger.error("base_dir is not a directory: {}", root)
        return []

    specs = pathspecs if pathspecs is not None else ["**/*"]
    spec = pathspec.PathSpec.from_lines("gitignore", specs)

    matched: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if spec.match_file(rel):
            matched.append(Path(str(path)))

    matched.sort()
    if not matched:
        logger.warning("No files matched pathspecs {} in {}", specs, root)
    return matched
