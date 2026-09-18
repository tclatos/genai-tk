"""Process-shared Ladybug database instances and extension management.

Ladybug is an embedded database: only one read-write ``Database`` object may
exist per file in a process. Multiple ``Connection``s created from that single
``Database`` may issue concurrent read and write transactions safely as long
as transactions touch disjoint rows.
"""

from __future__ import annotations

import atexit
import os
import re
import threading
from pathlib import Path
from typing import Any

from loguru import logger

_SHARED_DATABASES: dict[tuple[str, bool], Any] = {}
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


def _parse_buffer_pool_size(value: int | str) -> int:
    """Parse a buffer-pool size into bytes.

    Accepts an int (bytes) or a human string like ``"4GB"``, "512MB", "64KB".
    """
    if isinstance(value, int):
        return value
    text = value.strip().upper().replace(" ", "").replace("IB", "B")
    match = re.fullmatch(r"([0-9.]+)(B|KB|MB|GB|TB)?", text)
    if not match:
        raise ValueError(f"Invalid buffer pool size: {value!r} (use bytes, '4GB', '512MB', ...)")
    factor = {"": 1, "B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[match.group(2) or ""]
    return int(float(match.group(1)) * factor)


def get_shared_database(
    db_path: str,
    *,
    enable_multi_writes: bool = True,
    preload_extensions: tuple[str, ...] | list[str] = ("vector", "fts"),
    read_only: bool = False,
    buffer_pool_size: int | str | None = None,
) -> Any:
    """Return a process-shared ``ladybug.Database`` instance for *db_path*.

    Ensures only a single ``Database`` handle is opened per (file path,
    read-only) pair across all worker threads in the current process,
    pre-loading requested extensions on first initialization.

    Args:
        db_path: Database file path or ``:memory:``.
        enable_multi_writes: Enable multi-write support on the shared Database.
            Ignored (treated as False) when ``read_only`` is True.
        preload_extensions: Extensions to install and load on creation.
        read_only: Open the database read-only (no WAL, no checkpointing;
            several read-only handles may coexist for the same file).
        buffer_pool_size: Explicit buffer-pool size (bytes or like "4GB"). When
            None, read from ``LADYBUG_BUFFER_POOL_SIZE``; when neither is set,
            use the engine default (~80% of system memory, which can squeeze
            the host process on long concurrent runs).
    """
    import ladybug

    if buffer_pool_size is None:
        env_size = os.getenv("LADYBUG_BUFFER_POOL_SIZE", "").strip()
        buffer_pool_size = _parse_buffer_pool_size(env_size) if env_size else None
    else:
        buffer_pool_size = _parse_buffer_pool_size(buffer_pool_size)

    norm_path = db_path if db_path == ":memory:" else str(Path(db_path).resolve())
    key = (norm_path, read_only)
    with _DB_LOCK:
        db = _SHARED_DATABASES.get(key)
        if db is None:
            db = ladybug.Database(
                norm_path,
                buffer_pool_size=buffer_pool_size,
                enable_multi_writes=False if read_only else enable_multi_writes,
                read_only=read_only,
            )
            if preload_extensions:
                preload_ladybug_extensions(db, extensions=preload_extensions)
            _SHARED_DATABASES[key] = db
        return db


def clear_shared_database_cache() -> None:
    """Clear all cached shared ``ladybug.Database`` instances across the process."""
    with _DB_LOCK:
        _SHARED_DATABASES.clear()


def shutdown_shared_databases() -> None:
    """Close every shared Database and its connections in a safe order at exit.

    Interpreter-shutdown garbage collection frees Connection and Database
    native objects in arbitrary order, which corrupts the native heap
    (observed as "free(): invalid size" after a multi-threaded run). Closing
    every live connection first and each Database afterwards gives the native
    teardown the ordering it requires.
    """
    with _DB_LOCK:
        databases = list(_SHARED_DATABASES.values())
        _SHARED_DATABASES.clear()
    for db in databases:
        connections = list(getattr(db, "_connections", []) or [])
        for conn in connections:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        try:
            db.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


atexit.register(shutdown_shared_databases)
