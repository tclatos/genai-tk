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
        if not app_conf_path.exists():
            app_conf_path = Path("config/app_conf.yaml").absolute()
        assert app_conf_path.exists(), f"cannot find config file: '{app_conf_path}'"
        return OmegaConfig.create(app_conf_path)

    @staticmethod
    def create(app_conf_path: Path) -> OmegaConfig:
        config = OmegaConf.load(app_conf_path)
        assert isinstance(config, DictConfig)

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

        return OmegaConfig(root=config, selected_config=selected_config)  # type: ignore

    def select_config(self, config_name: str) -> None:
        """Select a different configuration section to override defaults."""
        if config_name not in self.root:
            logger.error(f"Configuration section '{config_name}' not found")
            raise ValueError(f"Configuration section '{config_name}' not found")
        logger.info(f"Switching to configuration section: {config_name}")
        self.selected_config = config_name

    def merge_with(self, file_path: str | UPath) -> OmegaConfig:
        """Merge additional YAML configuration file into the current config.

        Args:
            file_path: Path to YAML file to merge
        Returns:
            self for method chaining
        """
        path = UPath(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file to merge not found: {file_path}")

        new_conf = OmegaConf.load(path)
        assert isinstance(new_conf, DictConfig), f"Added conf not a Dict : type{new_conf}"

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
        """
        # Create merged config with runtime overrides first
        merged = OmegaConf.merge(self.root, self.selected or {})
        try:
            value = OmegaConf.select(merged, key)
            if value is None:
                if default is not None:
                    return default
                else:
                    raise ValueError(f"Configuration key '{key}' not found")
            return value
        except Exception as e:
            if default is not None:
                return default
            raise ValueError(f"Configuration key '{key}' not found") from e

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
        """Get a string configuration value."""
        value = self.get(key, default)
        if not isinstance(value, str):
            raise TypeError(f"Configuration value for '{key}' is not a string (its a {type(value)})")
        return value

    def get_bool(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean configuration value.

        Handles both native boolean values and string representations ('true', 'false', '1', '0', ...).
        """
        value = self.get(key, default)
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ("true", "1", "yes"):
                return True
            if value in ("false", "0", "no", "[]"):
                return False
            raise TypeError(f"Cannot convert string '{value}' to boolean for key '{key}'")
        if not isinstance(value, bool):
            raise TypeError(f"Configuration value for '{key}' is not a boolean (its a {type(value)})")
        return value

    def get_list(self, key: str, default: Optional[list] = None, value_type: type[T] | Any = Any) -> list[T]:
        """Get a list configuration value.

        Args:
            key: Configuration key in dot notation
            default: Default value if key not found
            type: Optional type to validate list elements against

        Returns:
            List of configuration values, optionally typed

        Example:
            ```python
            # Get untyped list
            modules = config.get_list("chains.modules")

            # Get typed list with validation
            names = config.get_list("user.names", type=str)
            ```
        """
        value = self.get(key, default)
        if not (isinstance(value, ListConfig) or isinstance(value, list)):
            raise TypeError(f"Configuration value for '{key}' is not a list (its a {type(value)})")

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
                    raise TypeError(
                        f"List item at index {i} for key '{key}' is not of type '{value_type}' but '{type(item)}' "
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
            ValueError: If value is not a dict or if expected keys validation fails
        """
        value = self.get(key)
        if not isinstance(value, DictConfig):
            raise TypeError(f"Configuration value for '{key}' is not a dict (its a {type(value)})")
        result = OmegaConf.to_container(value, resolve=True)
        if expected_keys is not None:
            missing_keys = [k for k in expected_keys if k not in result]
            if missing_keys:
                raise KeyError(f"Missing required keys '{key}': {', '.join(missing_keys)}")
        return result  # pyright: ignore[reportReturnType]

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> UPath:
        """Get a directory path. Can be local or remote  (https, S3, webdav, sftp,...)

        Args:
            key: Configuration key containing the path
            create_if_not_exists: If True, create directory when missing
        Returns:
            The Path object
        Raises:
            ValueError: If path doesn't exist, is not a directory, or create_if_not_exists=False
        """
        path = UPath(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory path for '{key}' does not exist: '{path}'")
        if not path.is_dir():
            raise FileNotFoundError(f"Path for '{key}' is not a directory: '{path}'")
        return path

    def get_file_path(self, key: str, check_if_exists: bool = True) -> UPath:
        """Get a file path. Can be local or remote  (https, S3, webdav, sftp,...)"""
        path = UPath(self.get_str(key))
        if not path.exists() and check_if_exists:
            raise FileNotFoundError(f"File path for '{key}' does not exist: '{path}'")
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
                temp_conf = OmegaConf.create({"_temp": file_path_raw})
                merged_for_resolution = OmegaConf.merge(resolution_config, temp_conf)
                file_path = str(merged_for_resolution["_temp"])
            else:
                file_path = str(file_path_raw)

            merge_path = Path(file_path)
            if not merge_path.exists():
                merge_path = Path("config") / file_path
            assert merge_path.exists(), f"cannot find config file: '{merge_path}'"

            merge_config = OmegaConf.load(merge_path)
            assert isinstance(merge_config, DictConfig), f"Merged config is not a DictConfig: {type(merge_config)}"

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

    The configuration value should be a string in the format 'module.submodule:function_or_class_name'.
    This method handles the import and returns the requested object.

    Examples:
        ```python
        #
        historical_price_func = import_from_qualified("src.ai_extra.smolagents_tools:get_historical_price")
        df = historical_price_func("AAPL", date(2024,1,1), date(2024,12,31))
        ```

    """
    if ":" not in qualified_name:
        raise ValueError(f"Invalid format for '{qualified_name}'. Expected format: 'module.submodule:object_name'")

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
