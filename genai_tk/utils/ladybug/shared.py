"""Process-shared Ladybug database instances and extension management.

Ladybug is an embedded database: only one read-write ``Database`` object may
exist per file in a process. Multiple ``Connection``s created from that single
``Database`` may issue concurrent read and write transactions safely as long
as transactions touch disjoint rows.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from loguru import logger

_SHARED_DATABASES: dict[str, Any] = {}
_DB_LOCK = threading.Lock()


def preload_ladybug_extensions(
    db_or_conn: Any,
    extensions: tuple[str, ...] | list[str] = ("vector", "fts"),
) -> None:
    """Install and load extensions on a Ladybug Database or Connection."""
    import ladybug

    conn = db_or_conn if isinstance(db_or_conn, ladybug.Connection) else ladybug.Connection(db_or_conn)
    for ext in extensions:
        try:
            conn.execute(f"INSTALL {ext}; LOAD EXTENSION {ext};")
        except Exception:
            try:
                conn.execute(f"LOAD EXTENSION {ext};")
            except Exception as exc:
                if "already loaded" not in str(exc).lower():
                    logger.debug("Could not load extension {}: {}", ext, exc)


def get_shared_database(
    db_path: str,
    *,
    enable_multi_writes: bool = True,
    preload_extensions: tuple[str, ...] | list[str] = ("vector", "fts"),
) -> Any:
    """Return a process-shared ``ladybug.Database`` instance for *db_path*.

    Ensures only a single ``Database`` handle is opened per file path across all
    worker threads in the current process, pre-loading requested extensions on
    first initialization.

    Args:
        db_path: Database file path or ``:memory:``.
        enable_multi_writes: Enable multi-write support on the shared Database.
        preload_extensions: Extensions to install and load on creation.
    """
    import ladybug

    norm_path = db_path if db_path == ":memory:" else str(Path(db_path).resolve())
    with _DB_LOCK:
        db = _SHARED_DATABASES.get(norm_path)
        if db is None:
            db = ladybug.Database(norm_path, enable_multi_writes=enable_multi_writes)
            if preload_extensions:
                preload_ladybug_extensions(db, extensions=preload_extensions)
            _SHARED_DATABASES[norm_path] = db
        return db


def clear_shared_database_cache() -> None:
    """Clear all cached shared ``ladybug.Database`` instances across the process."""
    with _DB_LOCK:
        _SHARED_DATABASES.clear()
