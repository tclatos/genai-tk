"""Backward-compatibility re-export module for LinkMLProcessor."""

from __future__ import annotations

from genai_tk.extra.structured.linkml_processor import (
    LinkMLProcessor,
    PydanticStructuredProcessor,
    extract_json_from_raw,
    linkml_extract,
    structured_extract,
    validate_model_capabilities,
)

__all__ = [
    "LinkMLProcessor",
    "PydanticStructuredProcessor",
    "extract_json_from_raw",
    "linkml_extract",
    "structured_extract",
    "validate_model_capabilities",
]
