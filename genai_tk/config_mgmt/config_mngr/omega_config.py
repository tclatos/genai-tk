"""OmegaConf-based application configuration manager."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated, Any, TypeVar

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from genai_tk.config_mgmt.config_exceptions import (
    ConfigFileNotFoundError,
    ConfigInterpolationError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
    ConfigValueError,
)
from genai_tk.config_mgmt.config_mngr import loading
from genai_tk.config_mgmt.config_mngr.proxy_bypass import apply_proxy_bypass
from genai_tk.utils.singleton import once

load_dotenv()

APPLICATION_CONFIG_FILE: str = "config/app_conf.yaml"

# Sentinel used to distinguish "no default provided" from "default=None".
_MISSING: Any = object()

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
"""Qualified name of any function - format: ``'module.path.function_name'``."""


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

        if "PWD" not in os.environ or "\\" in os.environ["PWD"]:
            os.environ["PWD"] = os.getcwd()

        # Process :env pseudo-key to load environment variables
        loading.process_env_variables(config)

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
            base_files = loading.resolve_merge_files(config, app_conf_path)
            # logger.debug(f"Loading {len(base_files)} YAML files from :merge: patterns")
            for yaml_path in base_files:
                config, provenance = loading.merge_file(config, yaml_path, provenance)

        # Apply :profile: inline block for the active profile
        config, provenance = loading.apply_profile_block(config, profile, app_conf_path, provenance)

        # Clean up pseudo-keys
        for key in [":merge", ":profile"]:
            if key in config:
                del config[key]

        # Apply the proxy bypass (NO_PROXY) derived from the merged config
        apply_proxy_bypass(config)

        instance = OmegaConfig(root=config, active_context=profile, provenance=provenance)  # type: ignore
        instance._validate_config()
        return instance

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

        loading.process_env_variables(new_conf, parent_config=self.root)

        # Remove pseudo-keys
        for key in [":merge", ":profile", ":env"]:
            if key in new_conf:
                del new_conf[key]

        loading.check_provenance_conflicts(new_conf, self.root, self.provenance, source)

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
        """
        # Create merged config with runtime overrides only if needed
        selected_ctx = self.selected
        merged = OmegaConf.merge(self.root, selected_ctx) if selected_ctx else self.root
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

    def get_str(self, key: str, default: str | None = None) -> str:
        """Get a string configuration value."""
        value = self.get(key, default)
        if value is None:
            return None  # type: ignore[return-value]
        if not isinstance(value, str):
            raise ConfigTypeError(key, expected_type=str, actual_type=type(value), actual_value=value)
        return value

    def get_bool(self, key: str, default: bool | None = None) -> bool:
        """Get a boolean configuration value.

        Handles both native boolean values and string representations ('true', 'false', '1', '0', ...).
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

    def get_list(self, key: str, default: list | None = None, value_type: type[T] | Any = Any) -> list[T]:
        """Get a list configuration value.

        Args:
            key: Configuration key in dot notation
            default: Default value if key not found
            value_type: Optional type to validate list elements against

        Returns:
            List of configuration values, optionally typed

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
        """
        path = Path(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ConfigFileNotFoundError(str(path))
        if not path.is_dir():
            raise ConfigValueError(key, value=str(path), reason="Path exists but is not a directory")
        return path

    def get_file_path(self, key: str, check_if_exists: bool = True) -> Path:
        """Get a file path.

        Args:
            key: Configuration key containing the file path
            check_if_exists: If True, verify that the file exists
        Returns:
            The Path object
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
