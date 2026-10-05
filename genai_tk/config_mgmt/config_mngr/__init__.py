"""Configuration manager using OmegaConf for YAML-based app configuration.

Public API:
- `global_config()` → OmegaConfig singleton with typed accessor methods
- `paths_config()` → PathsConfig (typed Pydantic model for the ``paths`` section)
- `get_raw_config()` → raw OmegaConf DictConfig for advanced operations
- Each module exposes its own `xxx_config()` accessor returning a typed Pydantic model
"""

from genai_tk.config_mgmt.config_mngr.omega_config import (
    APPLICATION_CONFIG_FILE,
    OmegaConfig,
    QualifiedCallable,
    QualifiedClassName,
    QualifiedFunctionName,
)
from genai_tk.config_mgmt.config_mngr.paths import PathsConfig, paths_config
from genai_tk.config_mgmt.config_mngr.runtime import (
    get_raw_config,
    global_config,
    global_config_reload,
    switch_profile,
    use_active_context,
)
from genai_tk.config_mgmt.config_mngr.yaml_loader import load_named_yaml_config, load_yaml_configs
from genai_tk.config_mgmt.import_utils import ImportResolver

__all__ = [
    "APPLICATION_CONFIG_FILE",
    "ImportResolver",
    "OmegaConfig",
    "PathsConfig",
    "QualifiedCallable",
    "QualifiedClassName",
    "QualifiedFunctionName",
    "get_raw_config",
    "global_config",
    "global_config_reload",
    "load_named_yaml_config",
    "load_yaml_configs",
    "paths_config",
    "switch_profile",
    "use_active_context",
]
