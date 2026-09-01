"""In-process parallelism for Ladybug backends sharing one ``Database``.

Ladybug is an embedded database: only one **read-write** ``Database`` object may
exist per file in a process, but multiple ``Connection``s created from that single
``Database`` may issue concurrent read *and* write transactions safely — the
transaction manager inside the shared ``Database`` serializes them correctly —
as long as the transactions touch **disjoint rows**.

Concurrent writes additionally require ``enable_multi_writes=True`` on the shared
``Database``; without it Ladybug rejects any second concurrent write transaction
with "Only one write transaction at a time is allowed".
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from genai_tk.utils.ladybug.shared import preload_ladybug_extensions

T = TypeVar("T")
R = TypeVar("R")


def _default_worker_factory(db: Any, num_threads: int) -> Any:
    """Instantiate a worker backend or Connection attached to *db*."""
    try:
        from genai_graph.kg.backend import LadybugBackend

        backend = LadybugBackend()
        backend.attach(db, num_threads=num_threads)
        return backend
    except Exception:
        import ladybug

        return ladybug.Connection(db, num_threads=num_threads)


class SharedLadybugParallel(BaseModel):
    """Run disjoint-row graph work concurrently against one shared Ladybug ``Database``.

    A Pydantic model managing a single shared ``ladybug.Database`` and a thread pool
    of worker connections/backends.
    """

    db_path: str
    max_workers: int = Field(default=4)
    num_threads_per_query: int = 0
    enable_multi_writes: bool = True
    preload_extensions: list[str] = Field(default_factory=lambda: ["vector", "fts"])
    worker_factory: Callable[[Any, int], Any] | None = None

    _db: Any = PrivateAttr(default=None)
    _pool: queue.Queue[Any] = PrivateAttr(default=None)
    _workers: list[Any] = PrivateAttr(default_factory=list)
    _closed: bool = PrivateAttr(default=False)
    _close_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("max_workers")
    @classmethod
    def _validate_max_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_workers must be >= 1, got {v}")
        return v

    def __init__(
        self,
        db_path: str | None = None,
        *,
        max_workers: int = 4,
        num_threads_per_query: int = 0,
        enable_multi_writes: bool = True,
        preload_extensions: list[str] | None = None,
        worker_factory: Callable[[Any, int], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if db_path is not None:
            kwargs["db_path"] = db_path
        if "max_workers" not in kwargs:
            kwargs["max_workers"] = max_workers
        if "num_threads_per_query" not in kwargs:
            kwargs["num_threads_per_query"] = num_threads_per_query
        if "enable_multi_writes" not in kwargs:
            kwargs["enable_multi_writes"] = enable_multi_writes
        if preload_extensions is not None:
            kwargs["preload_extensions"] = preload_extensions
        if worker_factory is not None:
            kwargs["worker_factory"] = worker_factory
        super().__init__(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the shared database and worker pool."""
        import ladybug

        self._db = ladybug.Database(self.db_path, enable_multi_writes=self.enable_multi_writes)
        if self.preload_extensions:
            preload_ladybug_extensions(self._db, extensions=self.preload_extensions)

        factory = self.worker_factory or _default_worker_factory
        self._workers = [factory(self._db, self.num_threads_per_query) for _ in range(self.max_workers)]
        self._pool = queue.Queue()
        for w in self._workers:
            self._pool.put(w)
        self._closed = False

    @property
    def primary(self) -> Any:
        """First worker instance, convenient for serial pre-reads before fan-out."""
        return self._workers[0] if self._workers else None

    def __enter__(self) -> SharedLadybugParallel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close worker connections and release the shared database."""
        with self._close_lock:
            if not self._closed:
                self._closed = True
                for w in self._workers:
                    try:
                        if hasattr(w, "conn") and w.conn is not None:
                            w.conn.close()
                        elif hasattr(w, "close"):
                            w.close()
                    except Exception:
                        pass
                    if hasattr(w, "db"):
                        w.db = None
                    if hasattr(w, "conn"):
                        w.conn = None
                try:
                    if hasattr(self._db, "close"):
                        self._db.close()
                except Exception:
                    pass
                self._workers = []
                self._db = None

    def map(self, func: Callable[[Any, T], R], items: list[T]) -> list[R | Exception]:
        """Apply ``func(worker_backend, item)`` over ``items`` concurrently in input order.

        Each running task borrows a worker from the pool and returns it on completion.
        If a worker raises, its slot holds the ``Exception`` instance rather than failing
        the whole batch.
        """
        results: list[R | Exception] = [None] * len(items)  # type: ignore[list-item]

        def _run(index: int, item: T) -> None:
            backend = self._pool.get()
            try:
                results[index] = func(backend, item)
            except Exception as exc:  # noqa: BLE001
                results[index] = exc
            finally:
                self._pool.put(backend)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run, i, item) for i, item in enumerate(items)]
            for fut in as_completed(futures):
                fut.result()

        return results


# Backward compatibility alias
SharedKuzuParallel = SharedLadybugParallel
