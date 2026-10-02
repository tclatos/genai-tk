"""Factory for instantiating document converters from configuration."""

from __future__ import annotations

from typing import Any

from loguru import logger

from genai_tk.config_mgmt.config_exceptions import ConfigError
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.import_utils import ImportResolver
from genai_tk.extra.markdownize.base import DocumentConverter

_CONVERTERS_SECTION = "markdownize_converters"


class ConverterFactory:
    """Factory for creating document converter instances defined in configuration."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DocumentConverter:
        """Instantiate a document converter by name with optional override parameters.

        The converter must be defined under the 'markdownize_converters' section
        of the configuration, e.g.:

        ```yaml
        markdownize_converters:
          mistral_ocr:
            class: genai_tk.extra.markdownize.mistral_ocr_converter.MistralOCRConverter
            params:
              model: mistral-ocr-latest
        ```

        Args:
            name: Converter name (e.g. 'markitdown', 'mistral_ocr', 'messy_xls_parser').
            **kwargs: Additional parameters passed to the converter constructor.

        Returns:
            Configured DocumentConverter instance.
        """
        section = cls._converters_section(name)
        cfg = section.get(name)
        if not isinstance(cfg, dict) or not cfg.get("class"):
            available = sorted(section.keys())
            raise KeyError(
                f"Unknown document converter '{name}'. Define it under '{_CONVERTERS_SECTION}' "
                f"in config/markdownize.yaml. Available converters: {available}"
            )

        params: dict[str, Any] = dict(cfg.get("params") or {})
        params.update(kwargs)
        params["name"] = name

        class_path = str(cfg["class"])
        logger.debug(f"Creating converter '{name}' using class {class_path}")
        converter_cls = ImportResolver.import_from_qualified(class_path)
        return converter_cls(**params)

    @classmethod
    def _converters_section(cls, name: str) -> dict[str, Any]:
        """Return the 'markdownize_converters' configuration section.

        Args:
            name: Converter name being resolved, used in error messages.

        Returns:
            The converter definitions keyed by name.
        """
        try:
            section = global_config().get_dict(_CONVERTERS_SECTION)
        except ConfigError as e:
            raise KeyError(
                f"Cannot create document converter '{name}': the '{_CONVERTERS_SECTION}' section "
                f"is not defined or the configuration file could not be loaded. ({e})"
            ) from e
        return section or {}
