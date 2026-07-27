"""LLM model selector widget for Streamlit applications.

Three-step selector: **Provider → Lab → Model** with compact capability
display and an optional thinking-effort slider for reasoning models.

Drop it into any Streamlit page::

    from genai_tk.webapp.ui_components.llm_selector import llm_selector_widget

    llm_selector_widget(st.sidebar)

It reads the current default from ``llm.models.default`` in the global config
and writes the selected model (with optional inline effort for thinking models)
back to the same key so all downstream code picks it up automatically.
"""

from __future__ import annotations

import re

from streamlit import session_state as sss
from streamlit.delta_generator import DeltaGenerator

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.factories.llm_factory import LlmFactory, LlmInfo
from genai_tk.core.providers import LAB_INFO, PROVIDER_INFO

# ── Constants ──────────────────────────────────────────────────────────────────

_ALL_LABS = "*"  # sentinel: show all labs / no filtering

_CAP_ICONS: dict[str, str] = {
    "vision": "👁",
    "thinking": "🧠",
    "structured_outputs": "📐",
    "pdf": "📄",
    "audio": "🎙",
    "video": "🎥",
}

_EFFORT_OPTIONS: list[str] = ["low", "medium", "high"]

# Inline effort pattern: "model (effort)@provider"
_EFFORT_RE = re.compile(r"^(?P<alias>.+?)\s*\((?P<effort>[A-Za-z0-9_-]+)\)\s*$")

# ── Formatting helpers ─────────────────────────────────────────────────────────


def _fmt_tokens(n: int | None) -> str:
    if not n:
        return ""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "?"
    return f"${c:.3f}" if c < 0.1 else f"${c:.2f}"


def _capability_line(info: LlmInfo) -> str:
    """One-line capability summary: context · cost · icons."""
    parts: list[str] = []
    ctx = info.effective_context_window
    out = info.effective_max_tokens
    if ctx:
        parts.append(f"{_fmt_tokens(ctx)} ctx")
    if out:
        parts.append(f"{_fmt_tokens(out)} out")
    p = info.profile
    if p is not None and (p.cost_input is not None or p.cost_output is not None):
        parts.append(f"{_fmt_cost(p.cost_input)}/{_fmt_cost(p.cost_output)}/Mtok")
    icons = [_CAP_ICONS[c] for c in info.effective_capabilities if c in _CAP_ICONS]
    if icons:
        parts.append(" ".join(icons))
    return "  ·  ".join(parts) if parts else ""


# ── Internal helpers ───────────────────────────────────────────────────────────


def _strip_effort(llm_id: str) -> tuple[str, str | None]:
    """Parse ``'model (effort)@provider'`` → ``('model@provider', 'effort')``."""
    if not llm_id:
        return llm_id, None
    model_part, sep, provider_part = llm_id.rpartition("@")
    if not sep:
        return llm_id, None
    m = _EFFORT_RE.match(model_part)
    if m:
        return f"{m.group('alias')}@{provider_part}", m.group("effort")
    return llm_id, None


def _provider_label(provider: str) -> str:
    info = PROVIDER_INFO.get(provider)
    gw = " (gateway)" if info and info.gateway else ""
    return f"{provider.replace('_', ' ').title()}{gw}"


def _lab_label(lab_key: str) -> str:
    if lab_key == _ALL_LABS:
        return "* All"
    lab = LAB_INFO.get(lab_key)
    return lab.display_name if lab else lab_key.title()


def _labs_for_provider(provider_models: list[LlmInfo]) -> list[str]:
    """Return sorted lab keys present in *provider_models*, with ``*`` first."""
    seen: set[str] = set()
    for info in provider_models:
        lab = info.effective_lab
        if lab:
            seen.add(lab)
    return [_ALL_LABS] + sorted(seen)


def _group_by_provider(models: dict[str, LlmInfo]) -> dict[str, list[LlmInfo]]:
    groups: dict[str, list[LlmInfo]] = {}
    for info in models.values():
        groups.setdefault(info.provider, []).append(info)
    return {p: sorted(ms, key=lambda m: m.model) for p, ms in groups.items()}


def _default_lab_for_provider(provider: str, provider_models: list[LlmInfo]) -> str:
    """Return the single lab if the provider is dedicated to one lab; else ``*``."""
    labs = _labs_for_provider(provider_models)
    # Remove the '*' sentinel, check if only one lab remains
    real_labs = [l for l in labs if l != _ALL_LABS]
    if len(real_labs) == 1:
        return real_labs[0]
    return _ALL_LABS


# ── Public helpers ─────────────────────────────────────────────────────────────


def set_active_llm(llm_id: str) -> None:
    """Force the Provider/Lab/Model widgets to reflect *llm_id* on this render.

    Resolves aliases/tags via :class:`LlmFactory`, persists the canonical id to
    ``llm.models.default``, and seeds the selector widgets' session-state keys
    so the selectboxes pick up the new value immediately — instead of only on
    the following rerun (Streamlit ignores ``index=`` once a widget key is
    already present in session state).

    Call this **before** :func:`llm_selector_widget` is rendered in the same
    script run (e.g. right after switching agent profile).

    Args:
        llm_id: LLM identifier, alias, or config tag (e.g. from an agent profile).
    """
    try:
        resolved = LlmFactory.resolve_llm_identifier(llm_id)
    except Exception:
        resolved = llm_id

    global_config().set("llm.models.default", resolved)

    base_llm, effort = _strip_effort(resolved)
    all_models = LlmFactory.known_items_dict()
    info = all_models.get(base_llm)
    if info is None and "@" in base_llm:
        model_part, _, prov_part = base_llm.rpartition("@")
        info = LlmInfo(id=base_llm, provider=prov_part, model=model_part)

    if info is not None:
        sss["sel_llm_provider"] = info.provider
        sss["_llm_prev_provider"] = info.provider  # prevents immediate lab reset
        sss["sel_llm_lab"] = info.effective_lab or _ALL_LABS
        sss["sel_llm_model"] = info.id

    if effort:
        sss["llm_thinking_effort"] = effort
        sss["sel_llm_effort"] = effort
    else:
        sss.pop("llm_thinking_effort", None)
        sss.pop("sel_llm_effort", None)


def current_llm_label() -> str:
    """Return the bare model name of the currently-selected LLM.

    Prefers the Model selectbox's session-state value, which already reflects
    the latest user interaction at the top of the script run — even before
    :func:`llm_selector_widget` re-renders — so callers can show an up-to-date
    label (e.g. an expander title) without lag. Falls back to the persisted
    config default when the widget hasn't been rendered yet.
    """
    model_id = sss.get("sel_llm_model") or global_config().get_str("llm.models.default") or ""
    base_llm, _ = _strip_effort(model_id)
    return base_llm.split("@")[0] if base_llm else ""


# ── Public widget ──────────────────────────────────────────────────────────────


def llm_selector_widget(w: DeltaGenerator) -> None:
    """Three-step LLM selector: **Provider → Lab → Model**.

    * **Provider**: only providers with available API key + installed module.
    * **Lab**: auto-filled for single-lab providers (e.g. ``anthropic``).  For
      gateway providers (e.g. ``openrouter``) shows all labs present in the DB
      plus ``*`` to see every model.
    * **Model**: filtered list with capabilities shown as a compact caption.
    * **Thinking effort**: slider (low / medium / high) for reasoning models;
      embedded inline as ``model (effort)@provider``.

    The selection is written to ``llm.models.default`` in the global config so
    all downstream factory calls pick it up.

    Args:
        w: Streamlit container to render into.
    """
    all_models: dict[str, LlmInfo] = LlmFactory.known_items_dict()
    if not all_models:
        w.warning("No LLM providers available.  Check that API keys are set.")
        return

    groups = _group_by_provider(all_models)
    sorted_providers = sorted(groups.keys())

    # ── Parse current default ──────────────────────────────────────────────
    # The persisted default may be a compact alias / config tag (e.g.
    # 'gpt_oss120@openrouter') rather than a canonical known_items_dict key,
    # so resolve it first. Without this, the raw-string lookup below silently
    # fails, current_info is None, and the widget falls back to whatever
    # provider happens to sort first alphabetically — showing an unrelated
    # model instead of the configured default.
    raw_current_llm: str = global_config().get_str("llm.models.default") or ""
    raw_base_llm, current_effort = _strip_effort(raw_current_llm)
    try:
        base_llm = LlmFactory.resolve_llm_identifier(raw_base_llm) if raw_base_llm else raw_base_llm
    except Exception:
        base_llm = raw_base_llm
    current_llm = base_llm
    current_info = all_models.get(base_llm)
    if current_info is None and "@" in base_llm:
        model_part, _, prov_part = base_llm.rpartition("@")
        if prov_part:
            current_info = LlmInfo(id=base_llm, provider=prov_part, model=model_part)
    current_provider = current_info.provider if current_info else sorted_providers[0]

    # ── Step 1: Provider ───────────────────────────────────────────────────
    # Initialise via session state only (no index= to avoid session-state conflict).
    if "sel_llm_provider" not in sss:
        sss["sel_llm_provider"] = current_provider
    if sss.get("sel_llm_provider") not in sorted_providers:
        sss["sel_llm_provider"] = current_provider

    # Provider and Lab rendered side-by-side
    col_prov, col_lab = w.columns(2)

    selected_provider: str = col_prov.selectbox(
        "Provider",
        sorted_providers,
        format_func=_provider_label,
        key="sel_llm_provider",
    )

    # ── Gather models for this provider ────────────────────────────────────
    provider_models = groups.get(selected_provider, [])
    prov_info = PROVIDER_INFO.get(selected_provider)

    # For OpenRouter (and other openrouter-catalog gateways), supplement the
    # YAML-only list with every model in the models.dev OpenRouter catalog so
    # the user can choose any model, not just explicitly configured ones.
    display_models: dict[str, LlmInfo] = dict(all_models)
    if prov_info and prov_info.gateway and selected_provider == "openrouter":
        from genai_tk.core.models_db import get_models_db

        db_catalog = get_models_db().provider_models("openrouter")
        for model_id in db_catalog:
            item_id = f"{model_id}@openrouter"
            if item_id not in display_models:
                display_models[item_id] = LlmInfo(id=item_id, provider="openrouter", model=model_id)
        provider_models = sorted(
            [m for m in display_models.values() if m.provider == "openrouter"],
            key=lambda m: m.model,
        )

    # ── Step 2: Lab ────────────────────────────────────────────────────────
    available_labs = _labs_for_provider(provider_models)
    auto_lab = _default_lab_for_provider(selected_provider, provider_models)

    # Desired lab: try to follow the currently-active model's lab, else auto.
    if current_info and current_info.provider == selected_provider:
        desired_lab = current_info.effective_lab or auto_lab
    else:
        desired_lab = auto_lab
    if desired_lab not in available_labs:
        desired_lab = _ALL_LABS

    # Reset lab session state when provider changes; never use index= to avoid
    # "widget created with default value AND set via Session State API" warning.
    if sss.get("_llm_prev_provider") != selected_provider:
        sss["_llm_prev_provider"] = selected_provider
        sss["sel_llm_lab"] = desired_lab
    elif "sel_llm_lab" not in sss:
        sss["sel_llm_lab"] = desired_lab
    elif sss["sel_llm_lab"] not in available_labs:
        sss["sel_llm_lab"] = desired_lab

    real_labs = [lab for lab in available_labs if lab != _ALL_LABS]
    # Gateway providers span multiple labs — never disable their lab selector.
    is_single_lab = len(real_labs) == 1 and not (prov_info and prov_info.gateway)

    selected_lab: str = col_lab.selectbox(
        "Lab",
        available_labs,
        format_func=_lab_label,
        key="sel_llm_lab",
        disabled=is_single_lab,
        help="Filter models by creator lab.  Select * to see all.",
    )
    if is_single_lab:
        selected_lab = real_labs[0]

    # ── Step 3: Model ──────────────────────────────────────────────────────
    if selected_lab == _ALL_LABS:
        filtered_models = provider_models
    else:
        filtered_models = [m for m in provider_models if m.effective_lab == selected_lab]
        if not filtered_models:
            filtered_models = provider_models  # fallback

    model_ids = [m.id for m in filtered_models]

    # Manage via session state only (no index=) to avoid the Streamlit warning
    # about a widget created with both a default value and Session State value.
    if sss.get("sel_llm_model") not in model_ids:
        sss["sel_llm_model"] = base_llm if base_llm in model_ids else (model_ids[0] if model_ids else "")

    selected_id: str = w.selectbox(
        "Model",
        model_ids,
        format_func=lambda mid: display_models[mid].model if mid in display_models else mid.split("@")[0],
        key="sel_llm_model",
    )

    selected_info = display_models.get(selected_id)
    if selected_info is None:
        model_part, _, prov_part = selected_id.rpartition("@")
        selected_info = LlmInfo(id=selected_id, provider=prov_part, model=model_part)

    # ── Capabilities caption ───────────────────────────────────────────────
    cap_line = _capability_line(selected_info)
    if cap_line:
        w.caption(cap_line)

    # ── Thinking effort slider ─────────────────────────────────────────────
    effort: str | None = None
    if selected_info.supports_thinking:
        # Manage via session state only (no value=) to avoid the Streamlit
        # warning about a widget created with both a default value and
        # Session State value.
        if sss.get("sel_llm_effort") not in _EFFORT_OPTIONS:
            sss["sel_llm_effort"] = current_effort if current_effort in _EFFORT_OPTIONS else "medium"
        effort = w.select_slider(
            "Thinking effort",
            options=_EFFORT_OPTIONS,
            key="sel_llm_effort",
        )
        sss["llm_thinking_effort"] = effort

    # ── Persist final ID ───────────────────────────────────────────────────
    if effort:
        model_part, _, prov_part = selected_id.rpartition("@")
        final_id = f"{model_part} ({effort})@{prov_part}"
    else:
        final_id = selected_id

    if final_id != current_llm:
        global_config().set("llm.models.default", final_id)
        w.success(f"✓ {final_id}")
