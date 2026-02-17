"""Configuration manager using OmegaConf for YAML-based app configuration.

Handles loading from app_conf.yaml with environment variable substitution and multiple
environments. Supports runtime overrides and implements singleton pattern.

Example:
```python
model = global_config().get("llm.models.default")
global_config().select_config("local")
global_config().set("llm.models.default", "gpt-4")
```
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from upath import UPath

from genai_tk.utils.config_exceptions import (
    ConfigFileNotFoundError,
    ConfigInterpolationError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.utils.singleton import once

load_dotenv()

APPLICATION_CONFIG_FILE: str = "config/app_conf.yaml"

T = TypeVar("T")


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
            raise ConfigParseError(str(app_conf_path), original_error=e)

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
            available = [k for k in self.root.keys() if not k.startswith(":")]
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

        # Check for LLM configuration
        try:
            llm_providers = self.get("llm.providers", default=None)
            if llm_providers is None:
                warnings.append("No LLM providers found (llm.providers). Ensure provider files are in the :merge list.")
            elif not isinstance(llm_providers, (list, ListConfig)):
                errors.append(f"llm.providers should be a list, got {type(llm_providers).__name__}")
            elif len(llm_providers) == 0:
                warnings.append("llm.providers is empty - no LLM models configured")
        except Exception as e:
            logger.debug(f"Could not validate llm.providers: {e}")

        # Check for embeddings configuration
        try:
            emb_providers = self.get("embeddings.providers", default=None)
            if emb_providers is None:
                warnings.append(
                    "No embeddings providers found (embeddings.providers). "
                    "Ensure provider files are in the :merge list."
                )
            elif not isinstance(emb_providers, (list, ListConfig)):
                errors.append(f"embeddings.providers should be a list, got {type(emb_providers).__name__}")
            elif len(emb_providers) == 0:
                warnings.append("embeddings.providers is empty - no embeddings models configured")
        except Exception as e:
            logger.debug(f"Could not validate embeddings.providers: {e}")

        # Check for default models
        try:
            default_llm = self.get("llm.models.default", default=None)
            if default_llm is None:
                warnings.append("No default LLM model configured (llm.models.default)")
        except Exception as e:
            logger.debug(f"Could not check default LLM: {e}")

        try:
            default_emb = self.get("embeddings.models.default", default=None)
            if default_emb is None:
                warnings.append("No default embeddings model configured (embeddings.models.default)")
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

    def merge_with(self, file_path: str | UPath) -> OmegaConfig:
        """Merge additional YAML configuration file into the current config.

        Args:
            file_path: Path to YAML file to merge
        Returns:
            self for method chaining
        """
        path = UPath(file_path)
        if not path.exists():
            raise ConfigFileNotFoundError(str(file_path))

        try:
            new_conf = OmegaConf.load(path)
        except Exception as e:
            raise ConfigParseError(str(file_path), original_error=e)

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
                            available = list(parent.keys())
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
                raise ConfigInterpolationError(key, str(e), original_error=e)
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

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> UPath:
        """Get a directory path. Can be local or remote  (https, S3, webdav, sftp,...)

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
        path = UPath(self.get_str(key))
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

    def get_file_path(self, key: str, check_if_exists: bool = True) -> UPath:
        """Get a file path. Can be local or remote  (https, S3, webdav, sftp,...)

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
        path = UPath(self.get_str(key))
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
                    str_value = str(merged_for_resolution["_temp"])
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
                    file_path = str(merged_for_resolution["_temp"])
                except Exception as e:
                    raise ConfigInterpolationError(":merge file path", str(file_path_raw), original_error=e)
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
                raise ConfigParseError(str(merge_path), original_error=e)

            if not isinstance(merge_config, DictConfig):
                raise ConfigTypeError(
                    f"merge_file_{merge_path.name}", expected_type="DictConfig", actual_type=type(merge_config)
                )

            # Process :env variables in the merged file (pass resolution config for interpolation)
            OmegaConfig._process_env_variables(merge_config, parent_config=resolution_config)

            # Recursively process :merge in the merged file (pass resolution config)
            merge_config = OmegaConfig._process_merge_files(merge_config, parent_config=resolution_config)

            # Clean up pseudo-keys before merging
            if ":merge" in merge_config:
                del merge_config[":merge"]
            if "merge" in merge_config:
                del merge_config["merge"]
            # :env keys are already removed during processing

            config = OmegaConf.merge(config, merge_config)

        return config


def global_config(reload: bool = False) -> OmegaConfig:
    """Get the global config singleton. Reload from file if 'reload" is True"""
    if reload:
        global_config_reload()
    return OmegaConfig.singleton()


def global_config_reload():
    """Invalidate the global config singleton value to make it reload from file"""
    OmegaConfig.singleton.invalidate()  # type: ignore


def import_from_qualified(qualified_name: str) -> Callable:
    """Dynamically import and return a function, class, or object by its qualified name.

    The configuration value can be:
    - Fully qualified: 'module.submodule:function_or_class_name'
    - Short class name: 'ClassName' (will search for the class in the project)

    If a short name is provided and multiple classes with that name exist, raises ValueError.

    Examples:
        ```python
        # Fully qualified name
        historical_price_func = import_from_qualified("src.ai_extra.smolagents_tools:get_historical_price")
        df = historical_price_func("AAPL", date(2024,1,1), date(2024,12,31))

        # Short class name (searches project automatically)
        subgraph_class = import_from_qualified("CrmExtractSubGraph")
        ```

    """
    if ":" not in qualified_name:
        # Short name provided - search for the class in the project
        return _import_from_short_name(qualified_name)

    module_path, object_name = qualified_name.split(":", 1)

    try:
        module = importlib.import_module(module_path)
        return getattr(module, object_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}' for {qualified_name}': {e}") from e
    except AttributeError as e:
        raise AttributeError(
            f"Cannot find object '{object_name}' in module '{module_path}' for {qualified_name}': {e}"
        ) from e


def _import_from_short_name(class_name: str) -> Callable:
    """Find and import a class by its short name by searching the project.

    Searches for Python files containing the class definition and imports it.
    Raises ValueError if zero or multiple matches are found.

    Args:
        class_name: The short class name to search for (e.g., "CrmExtractSubGraph")

    Returns:
        The imported class

    Raises:
        ValueError: If no class found or multiple classes with the same name exist
    """
    import ast
    from pathlib import Path

    # Determine the project root (look for common markers)
    current_dir = Path.cwd()
    project_root = current_dir

    # Look for project markers
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            project_root = parent
            break

    matches = []

    # Search for Python files in the project
    for py_file in project_root.rglob("*.py"):
        # Skip virtual environments and common excluded directories
        if any(part in py_file.parts for part in [".venv", "venv", "__pycache__", ".git", "node_modules", ".eggs"]):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Found a matching class
                    # Convert file path to module path
                    rel_path = py_file.relative_to(project_root)
                    module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
                    module_path = ".".join(module_parts)

                    matches.append((module_path, class_name, py_file))
                    break  # Only one class definition per file with this name
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed
            continue

    if len(matches) == 0:
        raise ValueError(
            f"No class named '{class_name}' found in the project. "
            f"Please use the fully qualified format 'module.path:ClassName'"
        )

    if len(matches) > 1:
        match_details = "\n".join([f"  - {m[0]}:{m[1]} (in {m[2]})" for m in matches])
        raise ValueError(
            f"Multiple classes named '{class_name}' found in the project:\n{match_details}\n"
            f"Please use the fully qualified format to specify which one you want."
        )

    # Single match found - import it
    module_path, object_name, _ = matches[0]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, object_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}' for class '{class_name}': {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Cannot find class '{object_name}' in module '{module_path}': {e}") from e


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
