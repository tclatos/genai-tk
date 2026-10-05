"""Global configuration singleton accessors."""

from __future__ import annotations

import os

from omegaconf import DictConfig

from genai_tk.config_mgmt.config_mngr.omega_config import OmegaConfig


def global_config(reload: bool = False) -> OmegaConfig:
    """Get the global config singleton. Reload from file if 'reload" is True"""
    if reload:
        global_config_reload()
    return OmegaConfig.singleton()


def global_config_reload() -> None:
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
