"""Monitoring backend toggle widget for Streamlit applications.

Provides a ``st.pills`` selector to enable/disable LangSmith and LangFuse
monitoring backends at runtime. Credential availability is checked from the
environment, and the config + singleton are updated without a full page reload.
After selection, clickable links to the active trace dashboards are displayed.
"""

from __future__ import annotations

import os

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from genai_tk.utils.tracing import MonitoringConfig, monitoring_config, reset_monitoring

# Human-readable labels and icons per backend
_BACKEND_META: dict[str, tuple[str, str]] = {
    "langsmith": ("LangSmith", "🔗"),
    "langfuse": ("LangFuse", "🪷"),
}

# Env vars whose presence indicates a backend is configured
_BACKEND_KEY_VARS: dict[str, list[str]] = {
    "langsmith": ["LANGSMITH_API_KEY"],
    "langfuse": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
}


def _backend_configured(backend: str) -> bool:
    """Return True if the required env-var(s) for *backend* are set."""
    for var in _BACKEND_KEY_VARS.get(backend, []):
        if not os.environ.get(var, "").strip():
            return False
    return True


def _langsmith_trace_url() -> str:
    """Return the LangSmith traces URL for the current project."""
    project = os.environ.get("LANGSMITH_PROJECT", "")
    base = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    # Derive the UI host from the API endpoint
    ui_host = base.replace("api.smith.langchain.com", "smith.langchain.com").rstrip("/")
    if project:
        return f"{ui_host}/o/projects/{project}/threads"
    return f"{ui_host}/o/projects"


def _langfuse_trace_url() -> str:
    """Return the LangFuse traces URL."""
    host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "http://localhost:3000"
    return f"{host.rstrip('/')}/traces"


_BACKEND_URL_FN = {
    "langsmith": _langsmith_trace_url,
    "langfuse": _langfuse_trace_url,
}


def monitoring_backend_pills(w: DeltaGenerator | None = None) -> list[str]:
    """Render a multi-select pills widget to toggle monitoring backends.

    Only LangSmith and LangFuse are exposed. A backend is disabled (⚠️ label)
    when its API credentials are absent; the pill can still be selected, but a
    warning is shown so the user knows what to configure.

    After the pills, clickable links to the active trace dashboards are
    displayed for each selected and configured backend.

    The active backend list is persisted to the global config and the
    monitoring singleton is reset so the next LLM call picks up the change.

    Args:
        w: Streamlit container to render into.  Defaults to ``st.sidebar``.

    Returns:
        The list of currently active backend keys (``["langsmith"]``, etc.).
    """
    container = w if w is not None else st.sidebar

    cfg: MonitoringConfig = monitoring_config()
    active_now: list[str] = list(cfg.backends)

    available_backends = list(_BACKEND_META.keys())

    def _label(b: str) -> str:
        icon, name = _BACKEND_META[b][1], _BACKEND_META[b][0]
        ok = _backend_configured(b)
        return f"{icon} {name}" if ok else f"{icon} {name} ⚠️"

    default_selection = [b for b in available_backends if b in active_now]

    selected = container.pills(
        "Monitoring",
        options=available_backends,
        format_func=_label,
        default=default_selection,
        selection_mode="multi",
        key="monitoring_backend_pills",
    )

    selected = selected or []

    # Warn about selected but unconfigured backends
    for b in selected:
        if not _backend_configured(b):
            vars_needed = ", ".join(f"`{v}`" for v in _BACKEND_KEY_VARS.get(b, []))
            container.warning(f"**{_BACKEND_META[b][0]}**: set {vars_needed} to enable tracing.", icon="⚠️")

    # Detect changes and apply them
    if set(selected) != set(active_now):
        try:
            from genai_tk.config_mgmt.config_mngr import global_config

            global_config().set("monitoring.backends", list(selected))
            reset_monitoring()
        except Exception:
            pass

    # ── Trace dashboard links ──────────────────────────────────────────────
    for b in selected:
        if _backend_configured(b):
            url_fn = _BACKEND_URL_FN.get(b)
            if url_fn:
                icon, name = _BACKEND_META[b][1], _BACKEND_META[b][0]
                container.link_button(f"{icon} Open {name} traces", url_fn(), use_container_width=True)

    return list(selected)
