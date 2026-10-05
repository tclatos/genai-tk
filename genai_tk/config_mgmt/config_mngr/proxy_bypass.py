"""Proxy-bypass application: merge YAML-configured hosts into ``NO_PROXY``.

The declarative host list lives in YAML — built-in defaults plus the
``net.proxy_bypass_hosts`` key (a list of hostnames). The environment variable
itself is never persisted: it is derived at config-load time by *merging* into
the current environment, so shell exports and ``~/.env`` entries are preserved
rather than shadowed.
"""

from __future__ import annotations

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def apply_proxy_bypass(config: DictConfig) -> None:
    """Merge the YAML-configured proxy bypass hosts into ``NO_PROXY``/``no_proxy``.

    Read inline from *config* (not via ``global_config()``) to avoid
    re-entrant singleton creation.
    """
    try:
        from genai_tk.utils.net_env import DEFAULT_BYPASS_HOSTS, ensure_no_proxy_hosts

        hosts = list(DEFAULT_BYPASS_HOSTS)
        extra = OmegaConf.select(config, "net.proxy_bypass_hosts", default=None) or []
        for item in extra:
            host = str(item).strip()
            if host and host not in hosts:
                hosts.append(host)
        ensure_no_proxy_hosts(hosts)
    except Exception as exc:  # best-effort: never block config loading
        logger.debug(f"Proxy bypass not applied: {exc}")
