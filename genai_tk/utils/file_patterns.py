"""File pattern matching utilities for resolving files based on include/exclude patterns.

Provides simple utilities to resolve file lists from root directories using glob patterns,
with support for include and exclude filters.
"""

from __future__ import annotations

import re

from loguru import logger
from upath import UPath


def resolve_config_path(path_str: str) -> str:
    """Resolve YAML config references like ${paths.data_root} in path strings.

    Args:
        path_str: Path string that may contain ${config.key} references

    Returns:
        Resolved path string with config references expanded

    Example:
        ```python
        resolve_config_path("${paths.data_root}/files")
        # Returns: "/home/user/data/files"
        ```
    """
    from genai_tk.utils.config_mngr import global_config

    pattern = r"\$\{([^}]+)\}"

    def replace_match(match: re.Match) -> str:
        config_key = match.group(1)
        try:
            cfg = global_config()
            resolved = cfg.get_str(config_key)
            if resolved is None:
                logger.warning(f"Config key '{config_key}' not found, keeping original reference")
                return match.group(0)
            return str(resolved)
        except Exception as e:
            logger.warning(f"Failed to resolve config key '{config_key}': {e}")
            return match.group(0)

    return re.sub(pattern, replace_match, path_str)


def resolve_files(
    root_dir: str,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    recursive: bool = False,
) -> list[UPath]:
    """Resolve list of files from root directory using include/exclude patterns.

    Args:
        root_dir: Root directory to search (supports config variables like ${paths.data_root})
        include_patterns: List of glob patterns to include (default: ["*.*"])
        exclude_patterns: List of glob patterns to exclude (default: None)
        recursive: Whether to search recursively

    Returns:
        List of resolved file paths

    Example:
        ```python
        # Find all markdown files
        files = resolve_files("/data", include_patterns=["*.md"])

        # Find specific patterns, excluding some
        files = resolve_files(
            "${paths.data_root}/docs",
            include_patterns=["report_*.md", "summary_*.md"],
            exclude_patterns=["*_draft.md"],
            recursive=True
        )
        ```
    """
    resolved_root = resolve_config_path(root_dir)
    if include_patterns is None:
        include_patterns = ["*.*"]

    root_path = UPath(resolved_root)
    if not root_path.exists():
        logger.error(f"Root directory does not exist: {root_path}")
        return []

    if not root_path.is_dir():
        logger.error(f"Root path is not a directory: {root_path}")
        return []

    # Collect all matching files
    matched_files: set[UPath] = set()

    for pattern in include_patterns:
        if recursive:
            matches = root_path.rglob(pattern)
        else:
            matches = root_path.glob(pattern)

        for match in matches:
            if match.is_file():
                matched_files.add(UPath(str(match)))

    if not matched_files:
        logger.warning(f"No files matched include patterns in {root_path}")
        return []

    # Apply exclude patterns
    if exclude_patterns:
        excluded_files: set[UPath] = set()
        for pattern in exclude_patterns:
            if recursive:
                matches = root_path.rglob(pattern)
            else:
                matches = root_path.glob(pattern)

            for match in matches:
                if match.is_file():
                    excluded_files.add(UPath(str(match)))

        matched_files = matched_files - excluded_files
        logger.info(f"Excluded {len(excluded_files)} files based on exclude patterns")

    result = sorted(matched_files)
    logger.info(f"Resolved {len(result)} files from {root_path}")
    return result
