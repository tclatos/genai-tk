"""Network-environment helpers — programmatic management of proxy bypasses.

Corporate proxies often whitelist only package hosts (pypi, github) and time
out on LLM API endpoints. Instead of hand-maintaining ``no_proxy`` exports in
shell startup files, compute per-host bypasses programmatically:

- :func:`ensure_no_proxy_hosts` merges hosts into ``NO_PROXY``/``no_proxy``
  in-process (idempotent).
- :func:`probe_host_reachable` checks whether a host is reachable with or
  without the ambient proxy.
- :func:`recommended_bypass_hosts` probes a host list and classifies each one.

The bypass host list is declarative: built-in defaults plus the
``net.proxy_bypass_hosts`` YAML key. It is applied automatically — merged into
``NO_PROXY``/``no_proxy`` when the config loads — and the ``cli info doctor``
command builds on these helpers: it diagnoses proxy problems and its ``--fix``
flag persists the computed bypass list to ``~/.env``.

Example:
    ```python
    from genai_tk.utils.net_env import ensure_no_proxy_hosts

    ensure_no_proxy_hosts(["localhost", "127.0.0.1"])
    ```
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import httpx
from loguru import logger

LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

# Hosts that must always bypass an HTTP proxy: loopback names plus the API
# endpoints genai-tk itself talks to (models.dev catalogue, common LLM gateways).
# Extend per project via the config key ``net.proxy_bypass_hosts``.
DEFAULT_BYPASS_HOSTS: tuple[str, ...] = (
    *LOOPBACK_HOSTS,
    "models.dev",
    "openrouter.ai",
    "api.deepinfra.com",
    "api.edenai.run",
    "api.eu.edenai.run",
    "api.smith.langchain.com",
)

HostStatus = Literal["proxy_ok", "bypass_needed", "unreachable"]
"""Reachability classification for one host:

- ``proxy_ok`` — reachable through the ambient proxy (no bypass needed).
- ``bypass_needed`` — blocked through the proxy but reachable directly;
  add it to ``NO_PROXY``.
- ``unreachable`` — blocked both ways (offline or firewalled).
"""


def default_bypass_hosts() -> list[str]:
    """Return the default bypass hosts plus any configured extras.

    Extra hosts come from the ``net.proxy_bypass_hosts`` config key (a list of
    hostnames), so projects can add their own API endpoints without code changes.
    """
    hosts = list(DEFAULT_BYPASS_HOSTS)
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        extra = global_config().get("net.proxy_bypass_hosts", []) or []
        hosts.extend(h for h in (str(item).strip() for item in extra) if h and h not in hosts)
    except Exception:  # config unavailable (tests, bare import) — defaults only
        pass
    return hosts


def ensure_no_proxy_hosts(hosts: Iterable[str], env: dict[str, str] | None = None) -> None:
    """Add *hosts* to ``NO_PROXY``/``no_proxy`` so they bypass any HTTP proxy.

    Idempotent: entries already present are kept, order is preserved, and new
    hosts are appended. Mutates ``os.environ`` unless *env* is given.

    Args:
        hosts: Hostnames to bypass (e.g. ``["localhost", "openrouter.ai"]``).
        env: Target mapping; defaults to ``os.environ``.
    """
    target = env if env is not None else os.environ
    for key in ("NO_PROXY", "no_proxy"):
        entries = [e.strip() for e in target.get(key, "").split(",") if e.strip()]
        entries.extend(host for host in hosts if host not in entries)
        target[key] = ",".join(entries)


def probe_host_reachable(host: str, *, use_env_proxy: bool, timeout: float = 3.0) -> bool:
    """Return True when an HTTPS request to *host* gets any HTTP response.

    Any status code counts as reachable — only connection/TLS failures mean
    the host is unreachable. Set ``use_env_proxy=False`` to probe directly,
    bypassing ``HTTP(S)_PROXY``/``NO_PROXY`` environment settings.

    Args:
        host: Hostname to probe (e.g. ``"models.dev"``).
        use_env_proxy: When True, honor proxy environment variables.
        timeout: Per-probe timeout in seconds.
    """
    url = f"https://{host}/"
    try:
        with httpx.Client(trust_env=use_env_proxy, timeout=timeout) as client:
            response = client.get(url)
            return response.status_code > 0
    except Exception as exc:
        logger.debug(f"Probe failed for {host} (use_env_proxy={use_env_proxy}): {exc}")
        return False


def classify_host(host: str, *, timeout: float = 3.0) -> HostStatus:
    """Probe one host through the proxy first, then directly.

    Args:
        host: Hostname to classify.
        timeout: Per-probe timeout in seconds.

    Returns:
        ``"proxy_ok"``, ``"bypass_needed"`` or ``"unreachable"``.
    """
    if probe_host_reachable(host, use_env_proxy=True, timeout=timeout):
        return "proxy_ok"
    if probe_host_reachable(host, use_env_proxy=False, timeout=timeout):
        return "bypass_needed"
    return "unreachable"


def recommended_bypass_hosts(
    hosts: Iterable[str] | None = None, *, timeout: float = 3.0
) -> dict[HostStatus, list[str]]:
    """Probe *hosts* in parallel and classify each one's proxy reachability.

    Args:
        hosts: Hostnames to probe; defaults to :func:`default_bypass_hosts`
            excluding loopback interfaces (which always bypass the proxy).
        timeout: Per-probe timeout in seconds.

    Returns:
        Mapping of ``"proxy_ok"``, ``"bypass_needed"`` and ``"unreachable"``
        to the corresponding host lists (``bypass_needed`` entries are the
        ones worth adding to ``NO_PROXY``).
    """
    raw_hosts = list(hosts) if hosts is not None else default_bypass_hosts()
    host_list = [h for h in raw_hosts if h not in LOOPBACK_HOSTS]
    result: dict[HostStatus, list[str]] = {"proxy_ok": [], "bypass_needed": [], "unreachable": []}
    if not host_list:
        return result
    with ThreadPoolExecutor(max_workers=min(8, len(host_list))) as pool:
        for host, status in zip(
            host_list, pool.map(lambda h: classify_host(h, timeout=timeout), host_list), strict=True
        ):
            result[status].append(host)
    return result


def no_proxy_entries(env: dict[str, str] | None = None) -> list[str]:
    """Return the current NO_PROXY/no_proxy entries (union of both spellings)."""
    target = env if env is not None else os.environ
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        for entry in target.get(key, "").split(","):
            entry = entry.strip()
            if entry and entry not in entries:
                entries.append(entry)
    return entries
