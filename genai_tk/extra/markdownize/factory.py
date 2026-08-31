"""Factory for instantiating document converters from configuration."""

from __future__ import annotations

from typing import Any

from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.import_utils import ImportResolver
from genai_tk.extra.markdownize.base import DocumentConverter

BUILTIN_CONVERTERS: dict[str, str] = {
    "markitdown": "genai_tk.extra.markdownize.markitdown_converter.MarkItDownConverter",
    "messy_xls": "genai_tk.extra.markdownize.excel_converter.MessyExcelConverter",
    "messy_xls_parser": "genai_tk.extra.markdownize.excel_converter.MessyExcelConverter",
    "edgeparse": "genai_tk.extra.markdownize.edgeparse_converter.EdgeParseConverter",
    "mistral_ocr": "genai_tk.extra.markdownize.mistral_ocr_converter.MistralOCRConverter",
    "mistral": "genai_tk.extra.markdownize.mistral_ocr_converter.MistralOCRConverter",
    "lighton_ocr": "genai_tk.extra.markdownize.lighton_ocr_converter.LightOnOCRConverter",
    "lighton": "genai_tk.extra.markdownize.lighton_ocr_converter.LightOnOCRConverter",
    "anydoc": "genai_tk.extra.markdownize.anydoc_converter.AnyDocConverter",
    "llm": "genai_tk.extra.markdownize.llm_converter.LLMConverter",
}


class ConverterFactory:
    """Factory for creating document converter instances by name or configuration."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DocumentConverter:
        """Instantiate a document converter by name with optional override parameters.

        Args:
            name: Converter name (e.g. 'markitdown', 'mistral_ocr', 'lighton_ocr', 'anydoc', 'llm').
            **kwargs: Additional parameters passed to the converter constructor.

        Returns:
            Configured DocumentConverter instance.
        """
        # 1. Check configuration under markdownize_converters
        class_path: str | None = None
        params: dict[str, Any] = {}

        try:
            cfg = global_config().get_dict(f"markdownize_converters.{name}")
            if cfg and isinstance(cfg, dict):
                class_path = cfg.get("class")
                params = dict(cfg.get("params", {}))
        except Exception:
            pass

        # 2. Fall back to built-in mapping
        if not class_path:
            if name in BUILTIN_CONVERTERS:
                class_path = BUILTIN_CONVERTERS[name]
            else:
                raise KeyError(
                    f"Unknown document converter '{name}'. Available built-in: {sorted(BUILTIN_CONVERTERS.keys())}"
                )

        params.update(kwargs)
        params["name"] = name

        logger.debug(f"Creating converter '{name}' using class {class_path}")
        converter_cls = ImportResolver.import_from_qualified(class_path)
        return converter_cls(**params)
