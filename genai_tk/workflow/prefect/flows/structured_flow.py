"""Backward-compatibility re-export module for linkml_flow."""

from __future__ import annotations

from genai_tk.workflow.prefect.flows.linkml_flow import (
    LinkMLExtractionManifest,
    LinkMLExtractionManifestEntry,
    StructuredExtractionManifest,
    StructuredExtractionManifestEntry,
    linkml_extraction_flow,
    linkml_single_input_flow,
    resolve_linkml_contract,
    resolve_structured_contract,
    structured_extraction_flow,
    structured_single_input_flow,
)

__all__ = [
    "LinkMLExtractionManifest",
    "LinkMLExtractionManifestEntry",
    "StructuredExtractionManifest",
    "StructuredExtractionManifestEntry",
    "linkml_extraction_flow",
    "linkml_single_input_flow",
    "resolve_linkml_contract",
    "resolve_structured_contract",
    "structured_extraction_flow",
    "structured_single_input_flow",
]
