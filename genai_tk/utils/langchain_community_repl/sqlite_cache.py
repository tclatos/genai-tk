"""Minimal SQLite-backed LLM cache — replaces ``langchain_community.cache.SQLiteCache``.

Implements the ``langchain_core.caches.BaseCache`` interface using the stdlib
``sqlite3`` module, so no third-party dependency is needed beyond ``langchain_core``.

Serialisation mirrors the original: each ``Generation`` object is stored as a
pickled blob. The schema is intentionally identical to the community version so
existing ``.db`` files remain compatible.
"""

from __future__ import annotations

import pickle
import sqlite3
import threading
from typing import Optional

from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS full_llm_cache (
    prompt  TEXT NOT NULL,
    llm     TEXT NOT NULL,
    idx     INTEGER NOT NULL,
    response BLOB NOT NULL
)
"""

_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS full_llm_cache_idx
ON full_llm_cache (prompt, llm, idx)
"""

RETURN_VAL_TYPE = list[Generation]


class SQLiteCache(BaseCache):
    """LangChain cache backed by a local SQLite database.

    Args:
        database_path: Path to the SQLite ``.db`` file.  Defaults to
            ``.langchain.db`` in the current working directory.
    """

    def __init__(self, database_path: str = ".langchain.db") -> None:
        self._path = database_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        with self._conn:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_INDEX)

    # ------------------------------------------------------------------
    # BaseCache interface
    # ------------------------------------------------------------------

    def lookup(self, prompt: str, llm_string: str) -> Optional[list[Generation]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT response FROM full_llm_cache WHERE prompt=? AND llm=? ORDER BY idx",
                (prompt, llm_string),
            )
            rows = cur.fetchall()
        if not rows:
            return None
        return [pickle.loads(row[0]) for row in rows]

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM full_llm_cache WHERE prompt=? AND llm=?",
                (prompt, llm_string),
            )
            self._conn.executemany(
                "INSERT INTO full_llm_cache (prompt, llm, idx, response) VALUES (?,?,?,?)",
                [(prompt, llm_string, i, pickle.dumps(gen)) for i, gen in enumerate(return_val)],
            )

    def clear(self, **kwargs) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM full_llm_cache")

    async def aclear(self, **kwargs) -> None:
        self.clear(**kwargs)
