"""Prefect runtime helpers for running flows with ephemeral or server settings.

This module centralises common Prefect setup used by CLI commands so that
flows can be executed with an in-process, ephemeral Prefect server without
requiring a long-lived API or agent.  Optionally, flows can connect to a
deployed Prefect server when ``prefect.api_url`` is configured in the
application config.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_LOGGING_LEVEL,
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
    PREFECT_SERVER_EPHEMERAL_ENABLED,
    temporary_settings,
)

T = TypeVar("T")


def _get_configured_api_url() -> str | None:
    """Return configured Prefect API URL from app config or env, if any."""
    # Explicit env var takes precedence
    env_url = os.environ.get("GENAI_PREFECT_API_URL")
    if env_url:
        return env_url

    # Check application config (best-effort)
    try:
        from genai_tk.utils.config_mngr import global_config

        return global_config().get("prefect.api_url", default=None)
    except Exception:
        return None


@contextmanager
def ephemeral_prefect_settings() -> Iterator[None]:
    """Temporarily configure Prefect to use an in-process ephemeral server.

    If ``prefect.api_url`` is set in the application config (or via the
    ``GENAI_PREFECT_API_URL`` environment variable), connects to the
    deployed Prefect server instead for full dashboard / history support.
    """
    configured_url = _get_configured_api_url()

    if configured_url:
        # Use the deployed Prefect server — just quiet down logging
        with temporary_settings({PREFECT_API_URL: configured_url, PREFECT_LOGGING_LEVEL: "WARNING"}):
            logging.getLogger("prefect").setLevel(logging.WARNING)
            logging.getLogger("prefect.flow_runs").setLevel(logging.WARNING)
            logging.getLogger("prefect.task_runs").setLevel(logging.WARNING)
            yield
        return

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
                PREFECT_LOGGING_LEVEL: "WARNING",  # Reduce Prefect logging noise
            }
        ):
            # Also quiet down Prefect loggers directly
            logging.getLogger("prefect").setLevel(logging.WARNING)
            logging.getLogger("prefect.flow_runs").setLevel(logging.WARNING)
            logging.getLogger("prefect.task_runs").setLevel(logging.WARNING)
            logging.getLogger("prefect.engine").setLevel(logging.ERROR)
            yield
    finally:
        # Restore previous environment variables.
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_flow_ephemeral(flow_fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a Prefect flow function inside an ephemeral (or configured server) runtime context."""
    with ephemeral_prefect_settings():
        return flow_fn(*args, **kwargs)
