"""Synchronise genai-tk model list with the deepagents-cli TOML provider config.

deepagents-cli reads ``~/.deepagents/config.toml`` to discover custom model
providers for its TUI ``/model`` switcher.  This module writes (or updates) a
``[models.providers.genai_tk]`` section that lists the identifiers from
``deepagent.switcher_models`` and points to
:class:`~genai_tk.agents.deepagent_cli.model_adapter.GenaiTkModelAdapter` as the
instantiation class.

The net effect: when the user presses ``/model`` in the TUI they see a
``genai_tk`` provider whose model list is exactly what was configured in
``config/basic/agents/deepagent.yaml``, rather than the full registry of every
installed LangChain provider package.

Example:
    ```python
    from genai_tk.agents.deepagent_cli.toml_bridge import write_genai_tk_provider

    models = ["default", "fast_model", "gpt41mini@openai"]
    write_genai_tk_provider(models)
    ```
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PROVIDER_NAME = "genai_tk"
_ADAPTER_CLASS_PATH = "genai_tk.agents.deepagent_cli.model_adapter:GenaiTkModelAdapter"
_CONFIG_PATH = Path.home() / ".deepagents" / "config.toml"


def write_genai_tk_provider(models: list[str], config_path: Path = _CONFIG_PATH) -> None:
    """Write or update the ``genai_tk`` provider block in the deepagents TOML.

    Reads the existing ``config.toml`` (creating it if absent), injects or
    replaces the ``[models.providers.genai_tk]`` block with *models*, then
    writes the file back.  After writing, the deepagents-cli model-availability
    cache is cleared so the next ``/model`` invocation picks up the changes.

    If *models* is empty this function is a no-op — the deepagents-cli default
    behaviour (show all installed provider models) is preserved.

    Args:
        models: genai-tk LLM identifiers to expose in the TUI ``/model`` picker.
        config_path: Override the default ``~/.deepagents/config.toml`` path
            (useful for tests).
    """
    if not models:
        return

    # --- Read existing config ------------------------------------------------
    existing: dict = {}
    if config_path.exists():
        try:
            existing = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # --- Merge our provider into the config dict ----------------------------
    models_section: dict = existing.setdefault("models", {})
    providers_section: dict = models_section.setdefault("providers", {})

    providers_section[_PROVIDER_NAME] = {
        "class_path": _ADAPTER_CLASS_PATH,
        # Point to an always-present env var so deepagents-cli's credential check
        # returns True.  Authentication is handled internally by LlmFactory;
        # this field is only needed to pass deepagents-cli's provider-recognition
        # guard in the /model hot-swap path.
        "api_key_env": "HOME",
        "models": list(models),
    }

    # --- Serialise back to TOML --------------------------------------------
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_to_toml(existing).strip() + "\n", encoding="utf-8")

    # --- Clear deepagents-cli cache so new models are visible immediately ---
    _clear_deepagents_cache()


def _clear_deepagents_cache() -> None:
    """Reset the cached model-availability dict inside deepagents-cli."""
    try:
        import deepagents_cli.model_config as _mc

        _mc._available_models_cache = None  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Minimal TOML serialiser (handles str, bool, int, float, list, dict)
# ---------------------------------------------------------------------------


def _to_toml(data: dict, _prefix: str = "") -> str:
    """Serialise a nested dict to a TOML string.

    Only handles the subset of TOML types needed for the deepagents config:
    str, bool, int, float, and list[str | int | float | bool].  Nested dicts
    become TOML sections (``[section.sub]``).  Intermediate section headers are
    only emitted when a dict level contains at least one scalar value — purely
    structural dicts (containing only sub-dicts) are silently skipped, which
    prevents spurious empty ``[models]`` / ``[models.providers]`` stanzas.
    """
    lines: list[str] = []
    deferred: list[tuple[str, dict]] = []

    for key, value in data.items():
        qualified = f"{_prefix}.{key}" if _prefix else key
        if isinstance(value, dict):
            deferred.append((qualified, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")

    # Emit scalars under the current section header (if any exist)
    body = "\n".join(lines)

    section_parts: list[str] = []
    for section_name, section_data in deferred:
        inner = _to_toml(section_data, _prefix=section_name)
        # Only write a section header when this dict level has direct scalars.
        # If it only contains sub-dicts the sub-sections carry their own headers.
        has_direct_scalars = any(not isinstance(v, dict) for v in section_data.values())
        if has_direct_scalars:
            section_parts.append(f"\n[{section_name}]\n{inner}")
        else:
            section_parts.append(f"\n{inner}")

    return body + "".join(section_parts)


def _toml_value(v: object) -> str:
    """Format a scalar or list value as a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(item) for item in v)
        return f"[{items}]"
    # Fallback: string representation
    return f'"{v!s}"'
