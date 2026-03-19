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
QualifiedCallable  # 'module.path:callable'  (generic)
QualifiedClassName  # 'module.path:ClassName'
QualifiedFunctionName  # 'module.path:function_name'
```
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Optional, TypeVar

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, StringConstraints

from genai_tk.utils.config_exceptions import (
    ConfigFileError,
    ConfigFileNotFoundError,
    ConfigInterpolationError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.utils.import_utils import (
    ImportResolver,
    get_module_from_qualified,
    get_object_name_from_qualified,
    import_from_qualified,
    split_qualified_name,
)
from genai_tk.utils.singleton import once

load_dotenv()

APPLICATION_CONFIG_FILE: str = "config/app_conf.yaml"

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Qualified callable type annotations
# ---------------------------------------------------------------------------
_QUALIFIED_PATTERN = r"^[\w.]+:[\w]+$"

QualifiedCallable = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified name of any callable - format: ``'module.path:callable'``."""

QualifiedClassName = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified class name -format: ``'module.path:ClassName'``."""

QualifiedFunctionName = Annotated[str, StringConstraints(pattern=_QUALIFIED_PATTERN)]
"""Qualified function name - format: ``'module.path:function_name'``."""


# ---------------------------------------------------------------------------
# PathsConfig schema (paths section of app_conf.yaml / baseline.yaml)
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Paths configuration section (``paths:`` in YAML).

    All path fields are validated ``Path`` objects. Use ``paths_config().project`` directly
    (returns ``Path`` object), or call ``.as_posix()`` for string representation.
    """

    home: DirectoryPath | None = Field(None, description="Home directory (HOME env var)")
    project: DirectoryPath = Field(..., description="Root of the project (PWD env var)")
    config: DirectoryPath = Field(..., description="Config directory (typically <project>/config/basic)")
    data_root: DirectoryPath = Field(..., description="Data root directory for caches, vector stores, etc.")
    data: DirectoryPath | None = Field(None, description="Alias for data_root (may be set separately)")
    models: DirectoryPath | None = Field(None, description="Models cache directory")

    model_config = ConfigDict(extra="allow")


class OmegaConfig(BaseModel):
    """Application configuration manager using OmegaConf."""

    root: DictConfig
    selected_config: str

    model_config = ConfigDict(arbitrary_types_allowed=True)  # to make pydantic happy

    @property
    def selected(self) -> DictConfig:
        return self.root.get(self.selected_config)

    @once
    def singleton() -> OmegaConfig:
        """Returns the singleton instance of Config."""

        # Load main config file
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

        # Load and merge additional config files
        config = OmegaConfig._process_merge_files(config)

        # Determine which config to use
        # @TODO : remove config_name_from_env as its set on the YAML
        config_name_from_env = os.environ.get("BLUEPRINT_CONFIG")
        config_name_from_yaml = config.get("default_config")  # type: ignore
        if config_name_from_env and config_name_from_env not in config:
            logger.warning(
                f"Configuration selected by environment variable 'BLUEPRINT_CONFIG' not found: {config_name_from_env}"
            )
            config_name_from_env = None
        if config_name_from_yaml and config_name_from_yaml not in config and config_name_from_yaml != "baseline":
            logger.warning(f"Configuration selected by key 'default_config' not found: {config_name_from_yaml}")
            config_name_from_yaml = None
        selected_config = config_name_from_env or config_name_from_yaml or "baseline"

        # Clean up pseudo-keys from final config
        if ":merge" in config:
            del config[":merge"]
        if "merge" in config:
            del config["merge"]
        # :env keys are already removed during processing

        # Perform early validation
        instance = OmegaConfig(root=config, selected_config=selected_config)  # type: ignore
        instance._validate_config()

        return instance

    def select_config(self, config_name: str) -> None:
        """Select a different configuration section to override defaults."""
        if config_name not in self.root:
            logger.error(f"Configuration section '{config_name}' not found")
            available = [str(k) for k in self.root.keys() if not str(k).startswith(":")]
            raise ConfigKeyNotFoundError(config_name, available_keys=available)
        logger.info(f"Switching to configuration section: {config_name}")
        self.selected_config = config_name

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
                    "No LLM providers found (llm.exceptions). Ensure provider files are in the :merge list."
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
                    "No embeddings providers found (embeddings.registry). Ensure provider files are in the :merge list."
                )
            elif not isinstance(emb_entries, (list, ListConfig)):
                errors.append(f"embeddings.registry should be a list, got {type(emb_entries).__name__}")
            elif len(emb_entries) == 0:
                warnings.append("embeddings.registry is empty - no embeddings models configured")
        except Exception as e:
            logger.debug(f"Could not validate embeddings: {e}")

        # Check for default models
        try:
            default_llm = self.get("llm.models.default", default=None)
            if default_llm is None or not str(default_llm).strip():
                errors.append("Missing required default LLM tag: llm.models.default")
        except Exception as e:
            logger.debug(f"Could not check default LLM: {e}")

        try:
            default_emb = self.get("embeddings.models.default", default=None)
            if default_emb is None or not str(default_emb).strip():
                errors.append("Missing required default embeddings tag: embeddings.models.default")
        except Exception as e:
            logger.debug(f"Could not check default embeddings: {e}")

        # Check for paths configuration
        try:
            paths = self.get("paths", default=None)
            if paths is None:
                errors.append("Missing required 'paths' configuration section")
            else:
                required_paths = ["project", "config"]
                for path_key in required_paths:
                    if not self.get(f"paths.{path_key}", default=None):
                        errors.append(f"Missing required path configuration: paths.{path_key}")
        except Exception as e:
            logger.debug(f"Could not validate paths: {e}")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        # Raise validation error if there are errors
        if errors:
            raise ConfigValidationError(errors, config_name=self.selected_config)

    def merge_with(self, file_path: str | Path) -> OmegaConfig:
        """Merge additional YAML configuration file into the current config.

        Args:
            file_path: Path to YAML file to merge
        Returns:
            self for method chaining
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigFileNotFoundError(str(file_path))

        try:
            new_conf = OmegaConf.load(path)
        except Exception as e:
            raise ConfigParseError(str(file_path), original_error=e) from e

        if not isinstance(new_conf, DictConfig):
            raise ConfigTypeError(f"merge_file_{path.name}", expected_type="DictConfig", actual_type=type(new_conf))

        # Process :env pseudo-key to load environment variables (pass self.root for interpolation)
        OmegaConfig._process_env_variables(new_conf, parent_config=self.root)

        # Process :merge pseudo-key recursively (pass self.root as parent config)
        new_conf = OmegaConfig._process_merge_files(new_conf, parent_config=self.root)

        # Clean up pseudo-keys before merging
        if ":merge" in new_conf:
            del new_conf[":merge"]
        if "merge" in new_conf:
            del new_conf["merge"]
        # :env keys are already removed during processing

        self.root = OmegaConf.merge(self.root, new_conf)  # type: ignore
        return self

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.models.default")
            default: Default value if key not found
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
                if default is not None:
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
            if default is not None:
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
        # Ensure the selected config section exists
        if self.selected_config not in self.root:
            self.root[self.selected_config] = OmegaConf.create({})

        # Get the selected config section (now guaranteed to exist)
        selected_section = self.root[self.selected_config]
        OmegaConf.update(selected_section, key, value, merge=True)

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get a string configuration value.

        Raises:
            ConfigKeyNotFoundError: If key not found and no default provided
            ConfigTypeError: If value is not a string
        """
        value = self.get(key, default)
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
            keys_to_process = [k for k in config.keys() if k not in [":env", ":merge", "merge"]]

        # Recursively process :env in nested sections
        for key in keys_to_process:
            if key in [":env", ":merge", "merge"]:
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

    @staticmethod
    def _process_merge_files(config: DictConfig, parent_config: DictConfig | None = None) -> DictConfig:
        """Process :merge pseudo-key to merge additional configuration files.

        Supports both old 'merge' key (with deprecation warning) and new ':merge' key.
        Allows recursive merging when merged files also contain :merge keys.

        Args:
            config: Configuration dictionary to process
            parent_config: Parent config for resolving interpolations in nested merges
        Returns:
            Merged configuration dictionary
        """
        # Check for deprecated 'merge' key
        if "merge" in config:
            logger.warning(
                "Deprecated: 'merge' key has been renamed to ':merge'. "
                "Please update your configuration file. Support for 'merge' will be removed in a future version."
            )
            merge_files_conf = config.get("merge", [])
        else:
            merge_files_conf = config.get(":merge", [])

        if not merge_files_conf:
            return config

        # Convert to list without resolving to avoid premature interpolation errors
        merge_files = OmegaConf.to_container(merge_files_conf, resolve=False)
        if not isinstance(merge_files, list):
            logger.warning(f":merge must be a list, got {type(merge_files)}")
            return config

        # Use parent_config or current config for interpolation resolution
        resolution_config = OmegaConf.merge(parent_config, config) if parent_config else config

        for file_path_raw in merge_files:
            # Resolve the file path with proper context
            if isinstance(file_path_raw, str) and "${" in file_path_raw:
                try:
                    temp_conf = OmegaConf.create({"_temp": file_path_raw})
                    merged_for_resolution = OmegaConf.merge(resolution_config, temp_conf)
                    if isinstance(merged_for_resolution, DictConfig):
                        file_path = str(merged_for_resolution.get("_temp"))
                    else:
                        file_path = str(file_path_raw)
                except Exception as e:
                    raise ConfigInterpolationError(":merge file path", str(file_path_raw), original_error=e) from e
            else:
                file_path = str(file_path_raw)

            merge_path = Path(file_path)
            searched_paths = [str(merge_path)]
            if not merge_path.exists():
                merge_path = Path("config") / file_path
                searched_paths.append(str(merge_path))

            if not merge_path.exists():
                raise ConfigFileNotFoundError(file_path, searched_paths)

            try:
                merge_config = OmegaConf.load(merge_path)
            except Exception as e:
                raise ConfigParseError(str(merge_path), original_error=e) from e

            if not isinstance(merge_config, DictConfig):
                raise ConfigTypeError(
                    f"merge_file_{merge_path.name}", expected_type="DictConfig", actual_type=type(merge_config)
                )

            # Process :env variables in the merged file (pass resolution config for interpolation)
            parent_resolution_config = resolution_config if isinstance(resolution_config, DictConfig) else None
            OmegaConfig._process_env_variables(merge_config, parent_config=parent_resolution_config)

            # Recursively process :merge in the merged file (pass resolution config)
            merge_config = OmegaConfig._process_merge_files(merge_config, parent_config=parent_resolution_config)

            # Clean up pseudo-keys before merging
            if ":merge" in merge_config:
                del merge_config[":merge"]
            if "merge" in merge_config:
                del merge_config["merge"]
            # :env keys are already removed during processing

            merged_config = OmegaConf.merge(config, merge_config)
            if not isinstance(merged_config, DictConfig):
                raise ConfigTypeError("merged_config", expected_type="DictConfig", actual_type=type(merged_config))
            config = merged_config

        return config


def global_config(reload: bool = False) -> OmegaConfig:
    """Get the global config singleton. Reload from file if 'reload" is True"""
    if reload:
        global_config_reload()
    return OmegaConfig.singleton()


def global_config_reload():
    """Invalidate the global config singleton value to make it reload from file"""
    OmegaConfig.singleton.invalidate()  # type: ignore


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


def select_active_config(config_name: str) -> None:
    """Deprecated: call ``global_config().select_config()`` directly."""
    global_config().select_config(config_name)


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
    "import_from_qualified",
    "split_qualified_name",
    "get_module_from_qualified",
    "get_object_name_from_qualified",
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


def load_yaml_configs(
    config_path: Path,
    top_level_key: str,
    *,
    list_merge_keys: list[str] | None = None,
) -> dict[str, Any] | list[Any]:
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

        # Directory (all *.yaml files merged, profiles lists concatenated)
        cfg = load_yaml_configs(
            Path("config/agents/deepagent"),
            "deepagent",
            list_merge_keys=["profiles"],
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

        # Overlay file onto global config so ${paths.*} interpolations resolve
        if base_cfg is not None:
            try:
                merged_node = OmegaConf.merge(base_cfg, file_node)
            except Exception:
                merged_node = file_node
        else:
            merged_node = file_node

        try:
            resolved: dict[str, Any] = OmegaConf.to_container(merged_node, resolve=True)  # type: ignore[assignment]
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

    return accumulated


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
    global_config().select_config("training_local")
    model = global_config().get("llm.models.default")
    print(model)
    print(global_config().get_list("cli.commands"))

    global_config().set("llm.models.default", "foo")
    print(global_config().get_str("llm.models.default"))

    global_config().select_config("training_openai")
    print(global_config().get("training_openai.dummy"))

    print(global_config().root.default_config)
    print(global_config().selected.llm.models.default)
