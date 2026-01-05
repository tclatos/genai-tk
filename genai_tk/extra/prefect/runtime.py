"""Prefect runtime helpers for running flows with ephemeral settings.

This module centralises common Prefect setup used by CLI commands so that
flows can be executed with an in-process, ephemeral Prefect server without
requiring a long‑lived API or agent.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
    PREFECT_SERVER_EPHEMERAL_ENABLED,
    temporary_settings,
)

T = TypeVar("T")


@contextmanager
def ephemeral_prefect_settings() -> Iterator[None]:
    """Temporarily configure Prefect to use an in-process ephemeral server.

    This helper disables any configured Prefect API URL for the duration
    of the context, enables the ephemeral server mode, and sanitises HTTP
    proxy environment variables that can interfere with localhost
    traffic. This ensures flows keep working even when PREFECT_API_URL
    is set but unreachable.
    """

    # Run with an ephemeral in-process server and ensure localhost
    # traffic is not sent through proxies.
    proxy_vars = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    )
    previous_env: dict[str, str | None] = {}

    for var in proxy_vars:
        previous_env[var] = os.environ.get(var)
        os.environ.pop(var, None)

    # Ensure localhost bypasses proxies.
    previous_env["NO_PROXY"] = os.environ.get("NO_PROXY")
    previous_env["no_proxy"] = os.environ.get("no_proxy")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    try:
        with temporary_settings(
            {
                PREFECT_API_URL: None,
                PREFECT_SERVER_EPHEMERAL_ENABLED: True,
                PREFECT_SERVER_ALLOW_EPHEMERAL_MODE: True,
            }
        ):
            yield
    finally:
        # Restore previous environment variables.
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_flow_ephemeral(flow_fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a Prefect flow function inside an ephemeral runtime context.

    The provided callable is expected to be a Prefect flow, but this helper
    does not depend on Prefect internals and simply executes the callable
    within :func:`ephemeral_prefect_settings`.
    """

    with ephemeral_prefect_settings():
        return flow_fn(*args, **kwargs)
