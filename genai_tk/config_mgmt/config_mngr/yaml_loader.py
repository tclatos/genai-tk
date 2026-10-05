"""Generic YAML config loader (file or directory)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, overload

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from genai_tk.config_mgmt.config_exceptions import (
    ConfigFileError,
    ConfigFileNotFoundError,
    ConfigInterpolationError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    yaml_config_validation,
)
from genai_tk.config_mgmt.config_mngr.runtime import get_raw_config

M = TypeVar("M", bound=BaseModel)


def _deep_merge_with_list_keys(base: dict, override: dict, list_keys: set[str]) -> dict:
    """Deep-merge two dicts; keys in *list_keys* are concatenated, not overwritten."""
    result = dict(base)
    for k, v in override.items():
        if k in list_keys and isinstance(v, list) and isinstance(result.get(k), list):
            result[k] = result[k] + v
        elif isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge_with_list_keys(result[k], v, list_keys)
        else:
            result[k] = v
    return result


@overload
def load_yaml_configs(
    config_path: Path,
    top_level_key: str,
    *,
    list_merge_keys: list[str] | None = None,
    model: None = None,
) -> dict[str, Any] | list[Any]: ...


@overload
def load_yaml_configs(
    config_path: Path,
    top_level_key: str,
    *,
    list_merge_keys: list[str] | None = None,
    model: type[M],
) -> M | list[M]: ...


def load_yaml_configs(
    config_path: Path,
    top_level_key: str,
    *,
    list_merge_keys: list[str] | None = None,
    model: type[M] | None = None,
) -> dict[str, Any] | list[Any] | M | list[M]:
    """Load configuration from a YAML file or a directory of YAML files.

    Supports OmegaConf ``${...}`` interpolations resolved against the global config
    (e.g. ``${paths.project}``, ``${paths.config}``).

    When *config_path* is a **directory**, all ``*.yaml`` / ``*.yml`` files are loaded
    in alphabetical order and merged:

    - If the top-level value is a **dict**: files are deep-merged.  Keys listed in
      *list_merge_keys* have their contents **concatenated** rather than overwritten
      (useful for ``profiles`` lists that span multiple files).
    - If the top-level value is a **list**: lists are concatenated.

    Files that do not contain *top_level_key* are silently skipped.

    Args:
        config_path: Path to a YAML file or a directory containing YAML files.
        top_level_key: Key at the top level of each YAML file whose value is returned.
        list_merge_keys: When merging dicts, these nested keys hold lists that should
            be concatenated across files.  Example: ``["profiles"]``.

    Returns:
        The merged / concatenated value under *top_level_key*.

    Example:
        ```python
        from genai_tk.config_mgmt.config_mngr import load_yaml_configs
        from pathlib import Path

        # Single file
        profiles = load_yaml_configs(Path("config/agents/deerflow.yaml"), "deerflow_agents")

        # Directory (all *.yaml files merged)
        cfg = load_yaml_configs(
            Path("config/agents/langchain"),
            "langchain_agents",
        )
        ```
    """
    list_keys: set[str] = set(list_merge_keys or [])

    if config_path.is_file():
        yaml_files = [config_path]
    elif config_path.is_dir():
        yaml_files = sorted([*config_path.glob("*.yaml"), *config_path.glob("*.yml")])
        if not yaml_files:
            raise ConfigFileError(
                str(config_path),
                "directory is empty — no *.yaml / *.yml files found",
                suggestion=f"Add at least one YAML file with a '{top_level_key}:' key to '{config_path}'.",
            )
    else:
        raise ConfigFileNotFoundError(str(config_path))

    # Load global config for OmegaConf interpolation; fail gracefully if unavailable
    try:
        base_cfg: DictConfig | None = get_raw_config()
    except Exception:
        base_cfg = None

    accumulated: dict[str, Any] | list[Any] | None = None

    for yaml_path in yaml_files:
        try:
            file_node = OmegaConf.load(yaml_path)
        except Exception as exc:
            raise ConfigParseError(str(yaml_path), original_error=exc) from exc

        # Overlay file onto global config so ${paths.*} interpolations resolve,
        # but strip top_level_key from base_cfg to prevent its pre-loaded values
        # from polluting the explicitly loaded file content.
        if base_cfg is not None:
            try:
                context_cfg = OmegaConf.masked_copy(base_cfg, [k for k in base_cfg.keys() if k != top_level_key])
                merged_node = OmegaConf.merge(context_cfg, file_node)
            except Exception:
                merged_node = file_node
        else:
            merged_node = file_node

        try:
            # Resolve only the specific section we need, not the full merged config.
            # This avoids spurious InterpolationKeyError from ${profile.*} placeholders
            # in other sections (e.g. workflows step inputs) that are only valid at
            # workflow execution time, not at config-load time.
            if top_level_key in merged_node:
                section = merged_node[top_level_key]
                resolved_value = OmegaConf.to_container(section, resolve=True)
                resolved: dict[str, Any] = {top_level_key: resolved_value}  # type: ignore[assignment]
            else:
                resolved = {}
        except Exception as exc:
            raise ConfigInterpolationError(
                key=str(yaml_path),
                interpolation=str(exc),
                original_error=exc,
            ) from exc

        if not isinstance(resolved, dict):
            raise ConfigTypeError(yaml_path.name, expected_type=dict, actual_type=type(resolved))

        if top_level_key not in resolved:
            logger.debug(f"Skipping '{yaml_path}': no '{top_level_key}' key found")
            continue

        value = resolved[top_level_key]

        if accumulated is None:
            accumulated = value
        elif isinstance(accumulated, list) and isinstance(value, list):
            accumulated = accumulated + value
        elif isinstance(accumulated, dict) and isinstance(value, dict):
            accumulated = _deep_merge_with_list_keys(accumulated, value, list_keys)
        else:
            raise ConfigTypeError(
                f"{top_level_key} in {yaml_path.name}",
                expected_type=type(accumulated).__name__,
                actual_type=type(value),
            )

    if accumulated is None:
        if config_path.is_dir():
            raise ConfigFileError(
                str(config_path),
                f"no file in the directory contains the '{top_level_key}' key",
                suggestion=f"Add a '{top_level_key}:' section to at least one YAML file in '{config_path}'.",
            )
        raise ConfigKeyNotFoundError(top_level_key)

    if model is not None:
        with yaml_config_validation(file_path=str(config_path), context=top_level_key):
            if isinstance(accumulated, dict):
                return model.model_validate(accumulated)
            return [model.model_validate(item) for item in accumulated]  # type: ignore[union-attr]

    return accumulated


def load_named_yaml_config(
    config_path: Path,
    top_level_key: str,
    name: str,
    model: type[M],
) -> M:
    """Load and validate a single named entry from a dict-keyed YAML section.

    Calls :func:`load_yaml_configs` to obtain the full section (a dict whose keys
    are entry names), looks up *name*, injects ``name`` into the raw dict when the
    key is absent, then validates against *model*.

    Args:
        config_path: Path to a YAML file or directory.
        top_level_key: Top-level YAML key whose value is a dict of named entries.
        name: Key to look up within that dict.
        model: Pydantic model class used for validation.

    Example:
        ```python
        config = load_named_yaml_config(Path("config/web_scrapers"), "web_scrapers", "my_scraper", WebScraperConfig)
        ```
    """
    entries: dict[str, Any] = load_yaml_configs(config_path, top_level_key)  # type: ignore[assignment]
    if not isinstance(entries, dict) or name not in entries:
        available = list(entries.keys()) if isinstance(entries, dict) else []
        raise KeyError(f"'{name}' not found under '{top_level_key}' in '{config_path}'. Available: {available}")
    raw: dict = entries[name]
    raw.setdefault("name", name)
    with yaml_config_validation(file_path=str(config_path), context=f"'{name}'"):
        return model.model_validate(raw)
