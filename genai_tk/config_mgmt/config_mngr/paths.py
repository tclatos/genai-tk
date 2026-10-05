"""Typed access to the ``paths`` configuration section."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, field_validator

from genai_tk.config_mgmt.config_exceptions import ConfigValidationError
from genai_tk.config_mgmt.config_mngr.runtime import global_config


class PathsConfig(BaseModel):
    """Typed ``paths`` configuration section with validated ``Path`` fields."""

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


def paths_config() -> PathsConfig:
    """Return typed paths configuration.

    Reads the ``paths`` section from the global config and validates it against
    ``PathsConfig``. All path fields are automatically validated as existing directories
    and returned as ``Path`` objects.

    Example:
        ```python
        from genai_tk.config_mgmt.config_mngr import paths_config

        project_dir = paths_config().project
        config_dir = paths_config().config
        ```
    """
    try:
        raw = global_config().get_dict("paths")
        return PathsConfig.model_validate(raw)
    except Exception as e:
        raise ConfigValidationError(
            [f"Invalid 'paths' configuration section: {e}"],
            config_name="paths",
        ) from e
