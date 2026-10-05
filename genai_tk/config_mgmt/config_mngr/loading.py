"""Config-loading internals: ``:merge:`` files, ``:profiles:`` blocks and ``:env:`` variables.

Module-level helpers implementing the load pipeline used by
``OmegaConfig.create`` and ``OmegaConfig.merge_yaml``:

- ``resolve_merge_files`` / ``merge_file`` — gitignore-style ``:merge:`` file loading
- ``apply_profile_block`` — inline ``:profiles:`` overlay for the active profile
- ``process_env_variables`` — ``:env:`` pseudo-key expansion into ``os.environ``
- ``check_provenance_conflicts`` — warn on overlapping leaf keys across files
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from genai_tk.config_mgmt.config_exceptions import ConfigParseError, ConfigTypeError


def build_gitignore_matcher(patterns: list[str]) -> Callable[[str], bool]:
    """Return a path matcher using pathspec when available, fnmatch otherwise."""
    try:
        import pathspec

        spec = pathspec.PathSpec.from_lines("gitignore", patterns)
        return spec.match_file
    except Exception as exc:
        logger.warning("pathspec unavailable for config merge ({}), using fnmatch fallback", exc)

        def _match(rel_path: str) -> bool:
            included = False
            for raw in patterns:
                is_negated = raw.startswith("!")
                pat = raw[1:] if is_negated else raw
                if fnmatch(rel_path, pat):
                    included = not is_negated
            return included

        return _match


def resolve_merge_files(config: DictConfig, app_conf_path: Path) -> list[Path]:
    """Resolve ``:merge:`` pathspec patterns to an ordered list of YAML files.

    Patterns are gitignore-style (pathspec gitignore) relative to the
    directory containing app_conf.yaml.  Lines starting with ``!`` exclude
    previously matched files.  app_conf.yaml itself is always skipped.
    """
    merge_raw = OmegaConf.select(config, ":merge", default=None)
    if merge_raw is None:
        return []

    patterns = OmegaConf.to_container(merge_raw, resolve=False)
    if not isinstance(patterns, list):
        patterns = [str(patterns)]

    base_dir = app_conf_path.parent.resolve()
    matcher = build_gitignore_matcher([str(p) for p in patterns])

    result: list[Path] = []
    for yaml_path in sorted(base_dir.rglob("*.yaml")):
        if yaml_path.resolve() == app_conf_path.resolve():
            continue
        rel = yaml_path.relative_to(base_dir)
        if matcher(str(rel)):
            result.append(yaml_path)
    return result


def apply_profile_block(
    config: DictConfig,
    profile: str,
    app_conf_path: Path,
    provenance: dict[str, list[Path]],
) -> tuple[DictConfig, dict[str, list[Path]]]:
    """Apply the active profile's inline ``:profiles:`` block.

    Load order within the profile block:

    1. Files matched by a nested ``:merge:`` key (if present).
    2. Remaining inline keys deep-merged on top.
    """
    profile_block = OmegaConf.select(config, ":profiles", default=None)
    if profile_block is None:
        return config, provenance

    if not isinstance(profile_block, DictConfig):
        logger.warning(":profiles: must be a dict keyed by profile name — ignored")
        return config, provenance

    profile_data = OmegaConf.select(profile_block, profile, default=None)
    if profile_data is None:
        available = list(profile_block.keys())
        logger.warning(f"Profile '{profile}' not found in :profiles: block. Available: {available}")
        return config, provenance

    if not isinstance(profile_data, DictConfig):
        logger.warning(f":profiles:{profile} must be a dict — ignored")
        return config, provenance

    # Handle nested :merge: within the profile block
    profile_merge_raw = OmegaConf.select(profile_data, ":merge", default=None)
    if profile_merge_raw is not None:
        patterns = OmegaConf.to_container(profile_merge_raw, resolve=False)
        if not isinstance(patterns, list):
            patterns = [str(patterns)]
        base_dir = app_conf_path.parent.resolve()
        matcher = build_gitignore_matcher([str(p) for p in patterns])
        for yaml_path in sorted(base_dir.rglob("*.yaml")):
            if yaml_path.resolve() == app_conf_path.resolve():
                continue
            rel = yaml_path.relative_to(base_dir)
            if matcher(str(rel)):
                config, provenance = merge_file(config, yaml_path, provenance)

    # Build profile overlay without pseudo-keys
    profile_dict = OmegaConf.to_container(profile_data, resolve=False)
    if isinstance(profile_dict, dict):
        profile_dict.pop(":merge", None)

    if profile_dict:
        profile_overlay = OmegaConf.create(profile_dict)
        process_env_variables(profile_overlay, parent_config=config)
        profile_source = Path(f":profiles:{profile}")
        for key in profile_overlay.keys():
            key_str = str(key)
            if not key_str.startswith(":"):
                provenance.setdefault(key_str, []).append(profile_source)
        merged = OmegaConf.merge(config, profile_overlay)
        if not isinstance(merged, DictConfig):
            raise ConfigTypeError("profile_overlay_merged", expected_type="DictConfig", actual_type=type(merged))
        config = merged

    return config, provenance


def merge_file(
    config: DictConfig,
    yaml_path: Path,
    provenance: dict[str, list[Path]],
) -> tuple[DictConfig, dict[str, list[Path]]]:
    """Load a single YAML file and merge it into config, updating provenance."""
    try:
        new_conf = OmegaConf.load(yaml_path)
    except Exception as e:
        raise ConfigParseError(str(yaml_path), original_error=e) from e

    if not isinstance(new_conf, DictConfig):
        raise ConfigTypeError(f"file_{yaml_path.name}", expected_type="DictConfig", actual_type=type(new_conf))

    process_env_variables(new_conf, parent_config=config)

    # Remove pseudo-keys (merged files cannot carry :merge/:profiles)
    for key in [":merge", ":profile", ":env"]:
        if key in new_conf:
            del new_conf[key]

    # Warn on overlapping sub-keys
    check_provenance_conflicts(new_conf, config, provenance, yaml_path)

    # Record provenance
    for key in new_conf.keys():
        key_str = str(key)
        if not key_str.startswith(":"):
            provenance.setdefault(key_str, []).append(yaml_path)

    merged = OmegaConf.merge(config, new_conf)
    if not isinstance(merged, DictConfig):
        raise ConfigTypeError("merged_config", expected_type="DictConfig", actual_type=type(merged))
    return merged, provenance


def check_provenance_conflicts(
    new_conf: DictConfig,
    current_config: DictConfig,
    provenance: dict[str, list[Path]],
    source: Path,
) -> None:
    """Warn when a new YAML file sets the same leaf key as an already-loaded file.

    Only warns on actual value collisions (non-dict leaves sharing the same key path),
    not on dict-valued sub-keys that different files legitimately extend.
    """
    for key in new_conf.keys():
        key_str = str(key)
        if key_str.startswith(":") or key_str not in provenance:
            continue
        try:
            existing_val = current_config.get(key_str)
            new_val = new_conf.get(key_str)
            if isinstance(existing_val, DictConfig) and isinstance(new_val, DictConfig):
                # Recurse one level: only report leaf-level conflicts
                leaf_conflicts = [
                    sub_key
                    for sub_key in set(existing_val.keys()) & set(new_val.keys())
                    if not isinstance(existing_val.get(sub_key), DictConfig)
                    or not isinstance(new_val.get(sub_key), DictConfig)
                ]
                if leaf_conflicts:
                    existing_files = [str(f) for f in provenance.get(key_str, [])]
                    logger.warning(
                        f"Config key '{key_str}' has overlapping leaf keys {sorted(leaf_conflicts)} "
                        f"defined in {existing_files} and {source}. "
                        "Later file wins. Either exclude one file in :merge: patterns (use '!' prefix) "
                        "or add a profile overlay in :profiles: to intentionally override values."
                    )
        except Exception:
            pass


def process_env_variables(config: DictConfig, parent_config: DictConfig | None = None) -> None:
    """Process the ``:env`` pseudo-key to load environment variables recursively.

    Processes ``:env`` keys at the root level and within nested configuration sections.
    """
    # Process :env at the current level
    process_env_at_level(config, parent_config)

    # Get list of keys to process (avoiding iteration issues)
    keys_to_process = []
    try:
        # Convert to container to get keys without triggering interpolation
        config_dict = OmegaConf.to_container(config, resolve=False)
        if isinstance(config_dict, dict):
            keys_to_process = list(config_dict.keys())
    except Exception:
        # If conversion fails, try direct iteration
        keys_to_process = [k for k in config.keys() if k not in [":env", ":merge", ":profiles"]]

    # Recursively process :env in nested sections
    for key in keys_to_process:
        if key in [":env", ":merge", ":profiles"]:
            continue
        try:
            value = config.get(key)
            if isinstance(value, DictConfig):
                process_env_variables(value, parent_config)
        except Exception:
            # Skip keys that cause issues
            continue


def process_env_at_level(config: DictConfig, parent_config: DictConfig | None = None) -> None:
    """Process the ``:env`` pseudo-key at a specific level."""
    env_vars = config.get(":env", {})
    if not env_vars:
        return

    if not isinstance(env_vars, (DictConfig, dict)):
        logger.warning(f":env must be a dictionary, got {type(env_vars)}")
        return

    # Convert to container without resolving to avoid premature interpolation errors
    env_dict = OmegaConf.to_container(env_vars, resolve=False)
    if not isinstance(env_dict, dict):
        logger.warning(f":env must be a dictionary, got {type(env_dict)} after conversion")
        return

    # If we have a parent config, merge it temporarily for interpolation resolution
    resolution_config = OmegaConf.merge(parent_config, config) if parent_config else config

    for var_name, var_value in env_dict.items():
        if not isinstance(var_name, str):
            logger.warning(f"Environment variable name must be a string, got {type(var_name)}: {var_name}")
            continue

        try:
            # Resolve any OmegaConf references in the value using the resolution config
            if isinstance(var_value, str) and "${" in var_value:
                # Create a temporary config with the interpolated value
                temp_conf = OmegaConf.create({"_temp": var_value})
                merged_for_resolution = OmegaConf.merge(resolution_config, temp_conf)
                if isinstance(merged_for_resolution, DictConfig):
                    str_value = str(merged_for_resolution.get("_temp"))
                else:
                    str_value = str(var_value)
            else:
                str_value = str(var_value)

            # Set environment variable
            os.environ[var_name] = str_value
        except Exception as e:
            logger.warning(f"Failed to resolve environment variable {var_name}: {e}")
            continue

    # Remove :env key after processing
    if ":env" in config:
        del config[":env"]
