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
    current_llm: str = global_config().get_str("llm.models.default") or ""
    base_llm, current_effort = _strip_effort(current_llm)
    current_info = all_models.get(base_llm)
    current_provider = current_info.provider if current_info else sorted_providers[0]

    # ── Step 1: Provider ───────────────────────────────────────────────────
    # Persist provider in session state separately so lab resets on provider change
    if "sel_llm_provider" not in sss:
        sss["sel_llm_provider"] = current_provider

    provider_idx = sorted_providers.index(current_provider) if current_provider in sorted_providers else 0
    selected_provider: str = w.selectbox(
        "Provider",
        sorted_providers,
        index=provider_idx,
        format_func=_provider_label,
        key="sel_llm_provider",
    )

    provider_models = groups[selected_provider]

    # ── Step 2: Lab ────────────────────────────────────────────────────────
    available_labs = _labs_for_provider(provider_models)

    # Auto-select: if provider changed or lab never set, default to single lab or *
    auto_lab = _default_lab_for_provider(selected_provider, provider_models)
    if "sel_llm_lab" not in sss:
        sss["sel_llm_lab"] = auto_lab

    # Determine initial lab selection
    init_lab: str
    if current_info and current_info.provider == selected_provider:
        init_lab = current_info.effective_lab or auto_lab
    else:
        init_lab = auto_lab
    if init_lab not in available_labs:
        init_lab = _ALL_LABS

    # For single-lab providers disable the widget (show as read-only)
    real_labs = [l for l in available_labs if l != _ALL_LABS]
    is_single_lab = len(real_labs) == 1

    selected_lab: str = w.selectbox(
        "Lab",
        available_labs,
        index=available_labs.index(init_lab),
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
    model_idx = model_ids.index(base_llm) if base_llm in model_ids else 0

    selected_id: str = w.selectbox(
        "Model",
        model_ids,
        index=model_idx,
        format_func=lambda mid: all_models[mid].model,
        key="sel_llm_model",
    )

    selected_info = all_models[selected_id]

    # ── Capabilities caption ───────────────────────────────────────────────
    cap_line = _capability_line(selected_info)
    if cap_line:
        w.caption(cap_line)

    # ── Thinking effort slider ─────────────────────────────────────────────
    effort: str | None = None
    if selected_info.supports_thinking:
        init_effort = current_effort if current_effort in _EFFORT_OPTIONS else "medium"
        sss.setdefault("llm_thinking_effort", init_effort)
        effort = w.select_slider(
            "Thinking effort",
            options=_EFFORT_OPTIONS,
            value=sss.get("llm_thinking_effort", init_effort),
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
