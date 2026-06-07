"""Configuration manager using OmegaConf for YAML-based app configuration.

Handles loading from app_conf.yaml with environment variable substitution and multiple
environments. Supports runtime overrides and implements singleton pattern.

Public API:
- `global_config()` → OmegaConfig singleton with typed accessor methods
- `paths_config()` → PathsConfig (typed Pydantic model for the ``paths`` section)
- `get_raw_config()` → raw OmegaConf DictConfig for advanced operations
- Each module exposes its own `xxx_config()` accessor returning a typed Pydantic model

QualifiedCallable type annotations for YAML fields holding qualified names:
```python
QualifiedCallable  # 'module.path.callable'  (generic)
QualifiedClassName  # 'module.path.ClassName'
QualifiedFunctionName  # 'module.path.function_name'
```
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated, Any, Optional, TypeVar, overload

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, StringConstraints, TypeAdapter, field_validator

from genai_tk.utils.config_exceptions import (
    ConfigFileError,
    ConfigFileNotFoundError,
    ConfigInterpolationError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
    yaml_config_validation,
)
from genai_tk.utils.import_utils import ImportResolver
from genai_tk.utils.singleton import once

# Sentinel used to distinguish "no default provided" from "default=None".
_MISSING: Any = object()

load_dotenv()

APPLICATION_CONFIG_FILE: str = "config/app_conf.yaml"

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

# ---------------------------------------------------------------------------
# Qualified callable type annotations
# ---------------------------------------------------------------------------
_QUALIFIED_PATTERN = r"^[\w]+([.][\w]+)+$"

QualifiedCallable = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified name of any callable - format: ``'module.path.callable'``."""

QualifiedClassName = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified class name - format: ``'module.path.ClassName'``."""

QualifiedFunctionName = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified function name - format: ``'module.path.function_name'``."""


# ---------------------------------------------------------------------------
# PathsConfig schema (paths section of app_conf.yaml / baseline.yaml)
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Paths configuration section (``paths:`` in YAML).

    All path fields are validated ``Path`` objects. Use ``paths_config().project`` directly
    (returns ``Path`` object), or call ``.as_posix()`` for string representation.

    The ``data_root`` directory is auto-created if it doesn't exist.
    """

    home: DirectoryPath | None = Field(None, description="Home directory (HOME env var)")
    project: DirectoryPath = Field(..., description="Root of the project (PWD env var)")
    config: DirectoryPath = Field(..., description="Config directory (typically <project>/config)")
    data_root: Path = Field(..., description="Data root directory for caches, vector stores, etc.")
    data: DirectoryPath | None = Field(None, description="Alias for data_root (may be set separately)")
    models: DirectoryPath | None = Field(None, description="Models cache directory")

    @field_validator("data_root", mode="after")
    @classmethod
    def ensure_data_root_exists(cls, v: Path) -> Path:
        """Auto-create data_root directory if it doesn't exist."""
        v = Path(v).expanduser().resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = ConfigDict(extra="allow")


class OmegaConfig(BaseModel):
    """Application configuration manager using OmegaConf."""

    root: DictConfig
    active_context: str
    provenance: dict[str, list[Path]] = Field(default_factory=dict)
    """Maps each top-level config key to the list of YAML files that contributed it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def selected(self) -> DictConfig:
        return self.root.get(self.active_context)

    @once
    def singleton() -> OmegaConfig:
        """Returns the singleton instance of Config."""

        app_conf_path = Path(APPLICATION_CONFIG_FILE)
        searched_paths = [str(app_conf_path)]
        if not app_conf_path.exists():
            app_conf_path = Path("config/app_conf.yaml").absolute()
            searched_paths.append(str(app_conf_path))

        if not app_conf_path.exists():
            raise ConfigFileNotFoundError(APPLICATION_CONFIG_FILE, searched_paths)

        return OmegaConfig.create(app_conf_path)

    @staticmethod
    def create(app_conf_path: Path) -> OmegaConfig:
        try:
            config = OmegaConf.load(app_conf_path)
        except Exception as e:
            raise ConfigParseError(str(app_conf_path), original_error=e) from e

        if not isinstance(config, DictConfig):
            raise ConfigTypeError("root", expected_type="DictConfig", actual_type=type(config), actual_value=config)

        os.environ["PWD"] = os.popen("pwd").read().strip()  # Hack because PWD is sometime set to a Windows path in WSL

        # Process :env pseudo-key to load environment variables
        OmegaConfig._process_env_variables(config)

        # Build initial provenance from app_conf itself
        provenance: dict[str, list[Path]] = {
            str(k): [app_conf_path] for k in config.keys() if not str(k).startswith(":")
        }

        # Determine profile name early (before merging)
        env_profile = os.environ.get("GENAITK_PROFILE")
        raw_profile = OmegaConf.select(config, "profile", default=None)
        if raw_profile is None:
            raw_profile = OmegaConf.select(config, "default_config", default="local")
        profile = env_profile or str(raw_profile)

        # Load files matched by :merge: pathspec patterns
        if ":merge" in config:
            base_files = OmegaConfig._resolve_merge_files(config, app_conf_path)
            # logger.debug(f"Loading {len(base_files)} YAML files from :merge: patterns")
            for yaml_path in base_files:
                config, provenance = OmegaConfig._merge_file(config, yaml_path, provenance)

        # Apply :profile: inline block for the active profile
        config, provenance = OmegaConfig._apply_profile_block(config, profile, app_conf_path, provenance)

        # Clean up pseudo-keys
        for key in [":merge", ":profile"]:
            if key in config:
                del config[key]

        instance = OmegaConfig(root=config, active_context=profile, provenance=provenance)  # type: ignore
        instance._validate_config()
        return instance

    # -----------------------------------------------------------------------
    # :merge: and :profiles: loading
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_merge_files(config: DictConfig, app_conf_path: Path) -> list[Path]:
        """Resolve :merge: pathspec patterns to an ordered list of YAML files.

        Patterns are gitignore-style (pathspec gitignore) relative to the
        directory containing app_conf.yaml.  Lines starting with ``!`` exclude
        previously matched files.  app_conf.yaml itself is always skipped.
        """
        import pathspec

        merge_raw = OmegaConf.select(config, ":merge", default=None)
        if merge_raw is None:
            return []

        patterns = OmegaConf.to_container(merge_raw, resolve=False)
        if not isinstance(patterns, list):
            patterns = [str(patterns)]

        base_dir = app_conf_path.parent.resolve()
        spec = pathspec.PathSpec.from_lines("gitignore", [str(p) for p in patterns])

        result: list[Path] = []
        for yaml_path in sorted(base_dir.rglob("*.yaml")):
            if yaml_path.resolve() == app_conf_path.resolve():
                continue
            rel = yaml_path.relative_to(base_dir)
            if spec.match_file(str(rel)):
                result.append(yaml_path)
        return result

    @staticmethod
    def _apply_profile_block(
        config: DictConfig,
        profile: str,
        app_conf_path: Path,
        provenance: dict[str, list[Path]],
    ) -> tuple[DictConfig, dict[str, list[Path]]]:
        """Apply the active profile's inline :profiles: block.

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
            import pathspec

            patterns = OmegaConf.to_container(profile_merge_raw, resolve=False)
            if not isinstance(patterns, list):
                patterns = [str(patterns)]
            base_dir = app_conf_path.parent.resolve()
            spec = pathspec.PathSpec.from_lines("gitignore", [str(p) for p in patterns])
            for yaml_path in sorted(base_dir.rglob("*.yaml")):
                if yaml_path.resolve() == app_conf_path.resolve():
                    continue
                rel = yaml_path.relative_to(base_dir)
                if spec.match_file(str(rel)):
                    config, provenance = OmegaConfig._merge_file(config, yaml_path, provenance)

        # Build profile overlay without pseudo-keys
        profile_dict = OmegaConf.to_container(profile_data, resolve=False)
        if isinstance(profile_dict, dict):
            profile_dict.pop(":merge", None)

        if profile_dict:
            profile_overlay = OmegaConf.create(profile_dict)
            OmegaConfig._process_env_variables(profile_overlay, parent_config=config)
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

    @staticmethod
    def _merge_file(
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

        OmegaConfig._process_env_variables(new_conf, parent_config=config)

        # Remove pseudo-keys (merged files cannot carry :merge/:profiles)
        for key in [":merge", ":profile", ":env"]:
            if key in new_conf:
                del new_conf[key]

        # Warn on overlapping sub-keys
        OmegaConfig._check_provenance_conflicts(new_conf, config, provenance, yaml_path)

        # Record provenance
        for key in new_conf.keys():
            key_str = str(key)
            if not key_str.startswith(":"):
                provenance.setdefault(key_str, []).append(yaml_path)

        merged = OmegaConf.merge(config, new_conf)
        if not isinstance(merged, DictConfig):
            raise ConfigTypeError("merged_config", expected_type="DictConfig", actual_type=type(merged))
        return merged, provenance

    @staticmethod
    def _check_provenance_conflicts(
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
                            "Later file wins. Use a profile overlay or overrides.yaml to override intentionally."
                        )
            except Exception:
                pass

    def use_context(self, context_name: str) -> None:
        """Activate a named context overlay. Values in the matching top-level key override defaults."""
        if context_name not in self.root:
            logger.error(f"Configuration context '{context_name}' not found")
            available = [str(k) for k in self.root.keys() if not str(k).startswith(":")]
            raise ConfigKeyNotFoundError(context_name, available_keys=available)
        logger.info(f"Switching to configuration context: {context_name}")
        self.active_context = context_name

    def _validate_config(self) -> None:
        """Perform early validation of configuration structure.

        Checks for common configuration issues and required keys to provide
        helpful error messages at startup rather than during execution.

        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        warnings = []

        # Check for LLM configuration (try new 'exceptions' key, fall back to legacy 'registry')
        try:
            llm_entries = self.get("llm.exceptions", default=None)
            if llm_entries is None:
                llm_entries = self.get("llm.registry", default=None)
            if llm_entries is None:
                warnings.append(
                    "No LLM providers found (llm.exceptions). Ensure provider YAML files are listed in :merge: patterns."
                )
            elif not isinstance(llm_entries, (list, ListConfig)):
                errors.append(f"llm.exceptions should be a list, got {type(llm_entries).__name__}")
            elif len(llm_entries) == 0:
                warnings.append("llm.exceptions is empty - no LLM exception models configured")
        except Exception as e:
            logger.debug(f"Could not validate llm: {e}")

        # Check for embeddings configuration
        try:
            emb_entries = self.get("embeddings.registry", default=None)
            if emb_entries is None:
                warnings.append(
                    "No embeddings providers found (embeddings.registry). Ensure provider YAML files are listed in :merge: patterns."
                )
            elif not isinstance(emb_entries, (list, ListConfig)):
                errors.append(f"embeddings.registry should be a list, got {type(emb_entries).__name__}")
            elif len(emb_entries) == 0:
                warnings.append("embeddings.registry is empty - no embeddings models configured")
        except Exception as e:
            logger.debug(f"Could not validate embeddings: {e}")

        # Check for default models
        try:
            default_llm = self.get("llm.models.default", default=_MISSING)
            if default_llm is not _MISSING and not str(default_llm).strip():
                errors.append("Missing required default LLM tag: llm.models.default")
        except Exception as e:
            logger.debug(f"Could not check default LLM: {e}")

        try:
            default_emb = self.get("embeddings.models.default", default=_MISSING)
            if default_emb is not _MISSING and not str(default_emb).strip():
                errors.append("Missing required default embeddings tag: embeddings.models.default")
        except Exception as e:
            logger.debug(f"Could not check default embeddings: {e}")

        # Check for paths configuration
        try:
            paths = self.get("paths", default=_MISSING)
            if paths is not _MISSING:
                required_paths = ["project", "config"]
                for path_key in required_paths:
                    val = self.get(f"paths.{path_key}", default=_MISSING)
                    if val is _MISSING or not val:
                        errors.append(f"Missing required path configuration: paths.{path_key}")
        except Exception as e:
            logger.debug(f"Could not validate paths: {e}")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        # Raise validation error if there are errors
        if errors:
            raise ConfigValidationError(errors, config_name=self.active_context)

    def merge_with(self, file_path: str | Path) -> OmegaConfig:
        """Merge a YAML file into the current config.

        Args:
            file_path: Path to YAML file to merge
        Returns:
            self for method chaining
        """
        return self.merge_yaml(file_path)

    def merge_yaml(self, content: str | Path) -> OmegaConfig:
        """Dynamically merge YAML content (raw string or file path) into the current config.

        When ``content`` is a string that resolves to an existing file path, the file is loaded.
        Otherwise the string is parsed as raw YAML.

        Args:
            content: YAML string, YAML file path (str or Path)
        Returns:
            self for method chaining

        Example:
            ```python
            cfg = global_config()
            # from a file
            cfg.merge_yaml(Path("config/extra.yaml"))
            # from a string
            cfg.merge_yaml("llm:\\n  models:\\n    default: gpt-4o@openai")
            ```
        """
        source: Path
        if isinstance(content, Path):
            if not content.exists():
                raise ConfigFileNotFoundError(str(content))
            new_conf = OmegaConf.load(content)
            source = content
        else:
            # Try as file path first
            candidate = Path(content)
            if candidate.exists():
                new_conf = OmegaConf.load(candidate)
                source = candidate
            else:
                # Parse as raw YAML string
                try:
                    new_conf = OmegaConf.load(io.StringIO(content))
                except Exception as e:
                    raise ConfigParseError("<string>", original_error=e) from e
                source = Path("<string>")

        if not isinstance(new_conf, DictConfig):
            raise ConfigTypeError(f"merge_yaml_{source.name}", expected_type="DictConfig", actual_type=type(new_conf))

        OmegaConfig._process_env_variables(new_conf, parent_config=self.root)

        # Remove pseudo-keys
        for key in [":merge", ":profile", ":env"]:
            if key in new_conf:
                del new_conf[key]

        OmegaConfig._check_provenance_conflicts(new_conf, self.root, self.provenance, source)

        for key in new_conf.keys():
            key_str = str(key)
            if not key_str.startswith(":"):
                self.provenance.setdefault(key_str, []).append(source)

        merged = OmegaConf.merge(self.root, new_conf)
        if not isinstance(merged, DictConfig):
            raise ConfigTypeError("merge_yaml_result", expected_type="DictConfig", actual_type=type(merged))
        self.root = merged  # type: ignore
        return self

    def config_keys_info(self) -> dict[str, list[str]]:
        """Return top-level config key provenance as a dict mapping key → list of source file names.

        Useful for introspection and the ``cli info config-keys`` command.

        Returns:
            Dict where each key is a top-level config key and the value is the list of
            source file paths (as strings) that contributed that key.
        """
        return {k: [str(p) for p in paths] for k, paths in self.provenance.items()}

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Get a configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.models.default")
            default: Default value if key not found. Pass ``None`` to return None
                when the key is absent without raising.
        Returns:
            The configuration value or default if not found
        Raises:
            ConfigKeyNotFoundError: If key not found and no default provided
            ConfigInterpolationError: If interpolation resolution fails
        """
        # Create merged config with runtime overrides first
        merged = OmegaConf.merge(self.root, self.selected or {})
        try:
            value = OmegaConf.select(merged, key)
            if value is None:
                if default is not _MISSING:
                    return default
                # Try to get available keys at the parent level for better error messages
                parts = key.split(".")
                if len(parts) > 1:
                    parent_key = ".".join(parts[:-1])
                    try:
                        parent = OmegaConf.select(merged, parent_key)
                        if isinstance(parent, DictConfig):
                            available = [str(k) for k in parent.keys()]
                            raise ConfigKeyNotFoundError(key, available_keys=available)
                    except Exception:
                        pass
                raise ConfigKeyNotFoundError(key)
            return value
        except ConfigKeyNotFoundError:
            raise
        except Exception as e:
            if default is not _MISSING:
                return default
            # Check if it's an interpolation error
            if "${" in str(e) or "interpolation" in str(e).lower():
                raise ConfigInterpolationError(key, str(e), original_error=e) from e
            raise ConfigKeyNotFoundError(key) from e

    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.models.default")
            value: Value to set
        """
        # Ensure the active context section exists
        if self.active_context not in self.root:
            self.root[self.active_context] = OmegaConf.create({})

        # Get the active context section (now guaranteed to exist)
        selected_section = self.root[self.active_context]
        OmegaConf.update(selected_section, key, value, merge=True)

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get a string configuration value.

        Raises:
            ConfigKeyNotFoundError: If key not found and no default provided
            ConfigTypeError: If value is not a string
        """
        value = self.get(key, default)
        if value is None:
            return None  # type: ignore[return-value]
        if not isinstance(value, str):
            raise ConfigTypeError(key, expected_type=str, actual_type=type(value), actual_value=value)
        return value

    def get_bool(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean configuration value.

        Handles both native boolean values and string representations ('true', 'false', '1', '0', ...).

        Raises:
            ConfigKeyNotFoundError: If key not found and no default provided
            ConfigTypeError: If value cannot be interpreted as boolean
        """
        value = self.get(key, default)
        if value is None:
            return None  # type: ignore[return-value]
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ("true", "1", "yes"):
                return True
            if value in ("false", "0", "no", "[]"):
                return False
            raise ConfigTypeError(
                key, expected_type="boolean or boolean-like string", actual_type=type(value), actual_value=value
            )
        if not isinstance(value, bool):
            raise ConfigTypeError(key, expected_type=bool, actual_type=type(value), actual_value=value)
        return value

    def get_list(self, key: str, default: Optional[list] = None, value_type: type[T] | Any = Any) -> list[T]:
        """Get a list configuration value.

        Args:
            key: Configuration key in dot notation
            default: Default value if key not found
            value_type: Optional type to validate list elements against

        Returns:
            List of configuration values, optionally typed

        Raises:
            ConfigKeyNotFoundError: If key not found and no default provided
            ConfigTypeError: If value is not a list or list elements don't match value_type

        Example:
            ```python
            # Get untyped list
            modules = config.get_list("chains.modules")

            # Get typed list with validation
            names = config.get_list("user.names", value_type=str)
            ```
        """
        value = self.get(key, default)
        if value is None:
            return None  # type: ignore[return-value]
        if not (isinstance(value, ListConfig) or isinstance(value, list)):
            raise ConfigTypeError(key, expected_type=list, actual_type=type(value), actual_value=value)

        # Handle both ListConfig and regular Python lists
        if isinstance(value, ListConfig):
            result = OmegaConf.to_container(value, resolve=True)
        else:
            result = value

        # Ensure result is a list
        if not isinstance(result, list):
            raise TypeError(f"Expected list for key '{key}' but got {type(result)}")

        # Type validation if type parameter is provided
        if value_type is not Any:
            for i, item in enumerate(result):
                if not isinstance(item, value_type):
                    raise ConfigTypeError(
                        f"{key}[{i}]", expected_type=value_type, actual_type=type(item), actual_value=item
                    )

        return result

    def get_dict(self, key: str, expected_keys: list | None = None) -> dict[str, Any]:
        """Get a dictionary configuration value.

        Args:
            key: Configuration key in dot notation
            expected_keys: Optional list of required keys to validate against
        Returns:
            The dictionary configuration value
        Raises:
            ConfigKeyNotFoundError: If key not found
            ConfigTypeError: If value is not a dict
            ConfigValidationError: If expected keys validation fails
        """
        value = self.get(key)
        if not isinstance(value, DictConfig):
            raise ConfigTypeError(key, expected_type=dict, actual_type=type(value), actual_value=value)
        result = OmegaConf.to_container(value, resolve=True)
        if expected_keys is not None:
            missing_keys = [k for k in expected_keys if k not in result]
            if missing_keys:
                errors = [f"Missing required key: '{k}'" for k in missing_keys]
                raise ConfigValidationError(errors, config_name=key)
        return result  # pyright: ignore[reportReturnType]

    # ------------------------------------------------------------------
    # Typed section accessors (Pydantic)
    # ------------------------------------------------------------------

    def section(self, key: str, model: type[M], *, default: M | None = None) -> M:
        """Load a top-level config section as a validated Pydantic model.

        Args:
            key: Configuration key in dot notation (e.g. ``"prefect"``).
            model: Pydantic model class to validate against.
            default: Returned when the section is missing or empty.
                When ``None`` the model is instantiated with no arguments
                (requires all fields to have defaults).

        Returns:
            Validated Pydantic model instance.
        """
        raw = self.get(key, default=None)
        if raw is None:
            if default is not None:
                return default
            return model.model_validate({})
        if isinstance(raw, (DictConfig, ListConfig)):
            raw = OmegaConf.to_container(raw, resolve=True)
        return model.model_validate(raw)

    def section_dict(self, key: str, model: type[M] | Any, *, inject_name: bool = True) -> dict[str, M]:
        """Load a top-level config section as a dict of named Pydantic models.

        Each sub-key becomes a dict entry validated against *model*.
        *model* can be a Pydantic ``BaseModel`` subclass or an ``Annotated``
        discriminated-union type (validated via ``TypeAdapter``).

        When *inject_name* is ``True`` (default), the dict key is injected
        as the ``name`` field if the model accepts it and the value is a dict
        without an explicit ``name``.

        Args:
            key: Configuration key in dot notation (e.g. ``"kv_store"``).
            model: Pydantic model class or Annotated union type for each entry.
            inject_name: Inject the dict key as ``name`` field.

        Returns:
            Dict of validated Pydantic model instances keyed by config name.
        """
        raw = self.get(key, default=None)
        if raw is None:
            return {}
        if isinstance(raw, (DictConfig, ListConfig)):
            raw = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(raw, dict):
            raise ConfigTypeError(key, expected_type=dict, actual_type=type(raw), actual_value=raw)

        # Use TypeAdapter for Annotated union types, model_validate for BaseModel
        use_adapter = not (isinstance(model, type) and issubclass(model, BaseModel))
        adapter = TypeAdapter(model) if use_adapter else None

        result: dict[str, M] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and inject_name and "name" not in v:
                v = {**v, "name": k}
            if adapter is not None:
                result[k] = adapter.validate_python(v)
            else:
                result[k] = model.model_validate(v)
        return result

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> Path:
        """Get a directory path.

        Args:
            key: Configuration key containing the path
            create_if_not_exists: If True, create directory when missing
        Returns:
            The Path object
        Raises:
            ConfigKeyNotFoundError: If key not found
            ConfigTypeError: If value is not a string
            ConfigFileNotFoundError: If path doesn't exist and create_if_not_exists=False
            ConfigValueError: If path exists but is not a directory
        """
        path = Path(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ConfigFileNotFoundError(str(path))
        if not path.is_dir():
            from genai_tk.utils.config_exceptions import ConfigValueError

            raise ConfigValueError(key, value=str(path), reason="Path exists but is not a directory")
        return path

    def get_file_path(self, key: str, check_if_exists: bool = True) -> Path:
        """Get a file path.

        Args:
            key: Configuration key containing the file path
            check_if_exists: If True, verify that the file exists
        Returns:
            The Path object
        Raises:
            ConfigKeyNotFoundError: If key not found
            ConfigTypeError: If value is not a string
            ConfigFileNotFoundError: If file doesn't exist and check_if_exists=True
        """
        path = Path(self.get_str(key))
        if not path.exists() and check_if_exists:
            raise ConfigFileNotFoundError(str(path))
        return path

    def get_dsn(self, key: str, driver: str | None = None) -> str:
        """Get a Database Source Name (DSN) compliant with SQLAlchemy URL format.
        The driver part of the connection can be changed (ex: postgress+"asyncpg")"""

        from genai_tk.utils.sql_utils import check_dsn_update_driver

        db_url = self.get_str(key)
        return check_dsn_update_driver(db_url, driver)

    @staticmethod
    def _process_env_variables(config: DictConfig, parent_config: DictConfig | None = None) -> None:
        """Process :env pseudo-key to load environment variables recursively.

        Processes :env keys at the root level and within nested configuration sections.

        Args:
            config: Configuration dictionary to process
            parent_config: Parent config for resolving interpolations in merged files
        """
        # Process :env at the current level
        OmegaConfig._process_env_at_level(config, parent_config)

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
                    OmegaConfig._process_env_variables(value, parent_config)
            except Exception:
                # Skip keys that cause issues
                continue

    @staticmethod
    def _process_env_at_level(config: DictConfig, parent_config: DictConfig | None = None) -> None:
        """Process :env pseudo-key at a specific level.

        Args:
            config: Configuration dictionary to process
            parent_config: Parent config for resolving interpolations in merged files
        """
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


def global_config(reload: bool = False) -> OmegaConfig:
    """Get the global config singleton. Reload from file if 'reload" is True"""
    if reload:
        global_config_reload()
    return OmegaConfig.singleton()


def global_config_reload():
    """Invalidate the global config singleton value to make it reload from file"""
    OmegaConfig.singleton.invalidate()  # type: ignore


def switch_profile(profile: str) -> OmegaConfig:
    """Switch to a different profile and reload the global config singleton.

    Sets ``GENAITK_PROFILE`` and reloads all config files from the new profile
    directory. Use this to switch between deployment environments (local, pytest,
    test_unit, prod) at runtime.

    Args:
        profile: Profile name matching a directory under ``config/profiles/<profile>/``.

    Returns:
        The newly loaded global config singleton.

    Example:
        ```python
        switch_profile("pytest")  # load config/profiles/pytest/
        switch_profile("local")  # back to default
        ```
    """
    os.environ["GENAITK_PROFILE"] = profile
    OmegaConfig.singleton.invalidate()  # type: ignore
    return OmegaConfig.singleton()


# ---------------------------------------------------------------------------
# Public typed API – preferred over calling global_config() directly
# ---------------------------------------------------------------------------


def paths_config() -> PathsConfig:
    """Return typed paths configuration.

    Reads the ``paths`` section from the global config and validates it against
    ``PathsConfig``. All path fields are automatically validated as existing directories
    and returned as ``Path`` objects.  Raises a ``ConfigValidationError`` with a descriptive
    message on missing or invalid fields.

    Example:
        ```python
        from genai_tk.utils.config_mngr import paths_config

        project_dir = paths_config().project
        config_dir = paths_config().config
        ```
    """
    from genai_tk.utils.config_exceptions import ConfigValidationError

    try:
        raw = global_config().get_dict("paths")
        return PathsConfig.model_validate(raw)
    except Exception as e:
        raise ConfigValidationError(
            [f"Invalid 'paths' configuration section: {e}"],
            config_name="paths",
        ) from e


def use_active_context(context_name: str) -> None:
    """Activate a named context overlay on the global config singleton."""
    global_config().use_context(context_name)


def get_raw_config() -> DictConfig:
    """Return the raw OmegaConf ``DictConfig`` for modules that need direct OmegaConf access.

    Only use this when you need to perform OmegaConf-level operations (e.g. merge,
    interpolation resolution).  Prefer typed accessors such as ``paths_config()``
    for normal application code.
    """
    return global_config().root


# ---------------------------------------------------------------------------
# Re-exported from import_utils for backward compatibility
# Import directly from genai_tk.utils.import_utils for new code.
# ---------------------------------------------------------------------------
__all__ = [
    "ImportResolver",
]


# ---------------------------------------------------------------------------
# Generic YAML config loader (file or directory)
# ---------------------------------------------------------------------------


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
        from genai_tk.utils.config_mngr import load_yaml_configs
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

    Returns:
        Validated model instance.

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


## for quick test ##
if __name__ == "__main__":
    # Get a config value
    model = global_config().get("llm.models.default")
    print(model)
    # Set a runtime override
    global_config().set("llm.models.default", "gpt-4")
    model = global_config().get("llm.models.default")
    print(model)
    print(global_config().get_list("cli.commands"))
    # Switch configurations
    global_config().use_context("training_local")
    model = global_config().get("llm.models.default")
    print(model)
    print(global_config().get_list("cli.commands"))

    global_config().set("llm.models.default", "foo")
    print(global_config().get_str("llm.models.default"))

    global_config().use_context("training_openai")
    print(global_config().get("training_openai.dummy"))

    print(global_config().root.default_config)
    print(global_config().selected.llm.models.default)
