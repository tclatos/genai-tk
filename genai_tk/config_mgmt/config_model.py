"""Base model for YAML-driven Pydantic configuration objects.

Provides ``from_yaml()``, ``from_config()``, and ``from_yaml_dict()`` classmethods
so that any config model can be hydrated from a YAML string, a file, a dict, or
directly from the global OmegaConf singleton.

Example:
    ```python
    from genai_tk.config_mgmt.config_model import ConfigModel


    class MyConfig(ConfigModel):
        host: str = "localhost"
        port: int = 8080


    # From a YAML string (great for tests)
    cfg = MyConfig.from_yaml("host: 0.0.0.0\\nport: 9090")

    # From the global config singleton
    cfg = MyConfig.from_config("my_section")

    # Dict-keyed (returns dict[str, MyConfig])
    items = MyConfig.from_yaml_dict(path, top_key="servers")
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel


class ConfigModel(BaseModel):
    """Base class for YAML-backed Pydantic configuration models.

    Subclass this instead of plain ``BaseModel`` to gain factory classmethods
    that streamline construction from YAML sources, OmegaConf config, or dicts.
    """

    # ------------------------------------------------------------------
    # from_yaml — tri-source factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        source: str | Path | dict[str, Any],
        *,
        top_key: str | None = None,
        resolve: bool = True,
    ) -> Self:
        """Construct an instance from a YAML string, file path, or dict.

        Args:
            source: One of:
                - ``str``: raw YAML text (parsed with ``yaml.safe_load``)
                - ``Path``: path to a ``.yaml`` file (loaded via OmegaConf)
                - ``dict``: already-parsed mapping
            top_key: If set, extract this key from the parsed dict before
                validating. Use when the model represents a sub-section of the
                YAML document.
            resolve: When ``True`` (default), OmegaConf interpolations such as
                ``${paths.data_root}`` are resolved against the global config.
                Set to ``False`` for standalone/test usage where no global config
                is available.

        Returns:
            Validated model instance.
        """
        raw = _parse_source(source, resolve=resolve)

        if top_key is not None:
            raw = raw.get(top_key, {})

        return cls.model_validate(raw)

    # ------------------------------------------------------------------
    # from_config — read from the global OmegaConf singleton
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, key: str) -> Self:
        """Construct an instance from the global config singleton.

        Equivalent to ``global_config().section(key, cls)``.

        Args:
            key: Dot-notation config key (e.g. ``"prefect"``).

        Returns:
            Validated model instance (empty-default when key is absent).
        """
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().section(key, cls)

    # ------------------------------------------------------------------
    # from_yaml_dict — dict-keyed factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml_dict(
        cls,
        source: str | Path | dict[str, Any],
        *,
        top_key: str | None = None,
        inject_name: bool = True,
        resolve: bool = True,
    ) -> dict[str, Self]:
        """Construct a dict of named instances from a YAML source.

        Each key in the source dict becomes an entry validated against this model.

        Args:
            source: YAML string, file path, or dict (same as ``from_yaml``).
            top_key: Extract this key from the parsed dict before iterating.
            inject_name: When ``True``, inject the dict key as the ``name`` field
                if the model has one and the entry doesn't already provide it.
            resolve: Resolve OmegaConf interpolations (see ``from_yaml``).

        Returns:
            Dict mapping entry names to validated model instances.
        """
        raw = _parse_source(source, resolve=resolve)

        if top_key is not None:
            raw = raw.get(top_key, {})

        if not isinstance(raw, dict):
            return {}

        result: dict[str, Self] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and inject_name and "name" not in v:
                v = {**v, "name": k}
            result[k] = cls.model_validate(v)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_source(source: str | Path | dict[str, Any], *, resolve: bool = True) -> dict[str, Any]:
    """Parse a tri-source input into a plain Python dict.

    When *resolve* is ``True`` and *source* is a Path, OmegaConf interpolations
    are resolved against the global config (merged context).
    """
    if isinstance(source, dict):
        return source

    if isinstance(source, Path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(source)

        if resolve:
            try:
                from genai_tk.config_mgmt.config_mngr import get_raw_config

                merged = OmegaConf.merge(get_raw_config(), cfg)
            except Exception:
                merged = cfg
        else:
            merged = cfg

        return OmegaConf.to_container(merged, resolve=resolve)  # type: ignore[return-value]

    if isinstance(source, str):
        import yaml as _yaml

        parsed = _yaml.safe_load(source)

        if resolve and parsed and isinstance(parsed, dict):
            try:
                from omegaconf import OmegaConf

                from genai_tk.config_mgmt.config_mngr import get_raw_config

                omega_node = OmegaConf.create(parsed)
                merged = OmegaConf.merge(get_raw_config(), omega_node)
                return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]
            except Exception:
                pass  # Fall through to unresolved dict

        return parsed if isinstance(parsed, dict) else {}

    raise TypeError(f"source must be str, Path, or dict — got {type(source).__name__}")


# TODO: Migrate Pattern A consumers to use ConfigModel:
#   - genai_tk/core/factories/llm_factory.py (_llm_section)
#   - genai_tk/core/factories/embeddings_factory.py (_embeddings_section)
#   - genai_tk/agents/tools/direct_browser/factory.py (_load_browser_config)
#   - genai_tk/agents/tools/sandbox_browser/factory.py (_load_browser_config)
