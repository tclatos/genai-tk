"""Shared fixtures for Prefect flow unit tests.

Spins up Prefect's official ephemeral test harness once per session so that
``@flow`` / ``@task`` callables run against an in-memory SQLite-backed server
instead of a real deployment.  The harness is the context manager
:func:`prefect.testing.utilities.prefect_test_harness` (not a pytest fixture),
applied here as a session-scoped autouse fixture for speed.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from prefect.testing.utilities import prefect_test_harness

# Bypass any forward proxy for all hosts during the test session: the only HTTP
# traffic these tests generate is to Prefect's ephemeral loopback test server,
# and a corporate/WSL ``HTTP_PROXY`` would otherwise break harness startup.
_NO_PROXY_VALUE = "*"


@pytest.fixture(scope="session", autouse=True)
def prefect_test_harness_session() -> Generator[None, None, None]:
    """Run all flow tests against an ephemeral in-memory Prefect server."""
    saved = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy", "PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW")}
    os.environ["NO_PROXY"] = _NO_PROXY_VALUE
    os.environ["no_proxy"] = _NO_PROXY_VALUE
    # The harness runs @task callables without a flow-run context, so the Prefect
    # API log handler would warn on every task log. This is the officially
    # recommended opt-out (see the warning's own guidance text).
    os.environ["PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW"] = "ignore"
    try:
        with prefect_test_harness():
            yield
    finally:
        for key, original in saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
