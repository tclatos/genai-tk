"""Proxy-bypass application: merge YAML-configured hosts into ``NO_PROXY``.

The declarative host list lives in YAML — built-in defaults plus the
``net.proxy_bypass_hosts`` key (a list of hostnames). The environment variable
itself is never persisted: it is derived at config-load time by *merging* into
the current environment, so shell exports and ``~/.env`` entries are preserved
rather than shadowed.
"""

from __future__ import annotations

import os

from loguru import logger
from omegaconf import DictConfig, OmegaConf

PROXY_ENV_VARS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "WSL_PAC_URL",
)


def purge_proxy_env(env: dict[str, str] | None = None) -> list[str]:
    """Remove ambient proxy environment variables from target mapping.

    Args:
        env: Target mapping to mutate; defaults to ``os.environ``.

    Returns:
        List of variable names that were removed.
    """
    target = env if env is not None else os.environ
    removed: list[str] = []
    for var in PROXY_ENV_VARS:
        if var in target:
            target.pop(var, None)
            removed.append(var)
    return removed


def apply_proxy_bypass(config: DictConfig, env: dict[str, str] | None = None) -> None:
    """Merge the YAML-configured proxy bypass hosts into ``NO_PROXY``/``no_proxy``.

    If ``net.disable_proxy`` is True in the configuration, all ambient proxy
    variables (``HTTP_PROXY``, ``HTTPS_PROXY``, etc.) are purged from the
    environment instead of configuring bypasses.

    Read inline from *config* (not via ``global_config()``) to avoid
    re-entrant singleton creation.

    Args:
        config: The loaded OmegaConf DictConfig.
        env: Target mapping to mutate; defaults to ``os.environ``.
    """
    target = env if env is not None else os.environ
    try:
        # Check if proxy is explicitly disabled in YAML config
        if bool(OmegaConf.select(config, "net.disable_proxy", default=False)):
            purged = purge_proxy_env(target)
            if purged:
                logger.debug(f"Proxy environment variables purged via net.disable_proxy: {', '.join(purged)}")
            return

        from genai_tk.utils.net_env import DEFAULT_BYPASS_HOSTS, ensure_no_proxy_hosts

        hosts = list(DEFAULT_BYPASS_HOSTS)
        extra = OmegaConf.select(config, "net.proxy_bypass_hosts", default=None) or []
        for item in extra:
            host = str(item).strip()
            if host and host not in hosts:
                hosts.append(host)
        ensure_no_proxy_hosts(hosts, env=target)
    except Exception as exc:  # best-effort: never block config loading
        logger.debug(f"Proxy bypass not applied: {exc}")
