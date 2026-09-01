"""Unit tests for genai_tk.utils.ladybug utilities."""

from __future__ import annotations

import pytest

from genai_tk.config_mgmt.features import is_available
from genai_tk.utils.ladybug import (
    SharedKuzuParallel,
    SharedLadybugParallel,
    clear_shared_database_cache,
)


def test_shared_ladybug_parallel_class_properties() -> None:
    assert SharedKuzuParallel is SharedLadybugParallel


def test_clear_shared_database_cache() -> None:
    clear_shared_database_cache()


@pytest.mark.skipif(not is_available("ladybug"), reason="ladybug not installed")
def test_shared_ladybug_parallel_instantiation() -> None:
    parallel = SharedLadybugParallel(
        db_path=":memory:",
        max_workers=2,
        preload_extensions=[],
        worker_factory=lambda db, threads: object(),
    )
    assert parallel.db_path == ":memory:"
    assert parallel.max_workers == 2
    assert parallel.primary is not None


@pytest.mark.skipif(not is_available("ladybug"), reason="ladybug not installed")
def test_shared_ladybug_parallel_invalid_workers() -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        SharedLadybugParallel(
            db_path=":memory:",
            max_workers=0,
            preload_extensions=[],
            worker_factory=lambda db, threads: object(),
        )
