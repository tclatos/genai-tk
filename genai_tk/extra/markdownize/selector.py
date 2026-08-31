"""Pathspec-driven converter selection and routing rules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PptConverter = Literal["via_pdf", "markitdown", "anydoc", "lighton_ocr", "llm"]
DocConverter = Literal["via_pdf", "markitdown", "anydoc", "lighton_ocr", "llm"]
ExcelConverter = Literal["via_pdf", "markitdown", "messy_xls_parser", "messy_xls", "anydoc", "llm"]
PdfConverter = Literal["mistral", "mistral_ocr", "markitdown", "edgeparse", "lighton", "lighton_ocr", "anydoc", "llm"]

MD_EXTENSIONS = {".md", ".markdown"}


def _expand_pattern(pattern: str) -> list[str]:
    """Expand shell-style brace alternatives into separate pathspec patterns."""
    match = re.search(r"\{([^}]+)\}", pattern)
    if match:
        alternatives = [s.strip() for s in match.group(1).split(",")]
        results = []
        for alt in alternatives:
            results.extend(_expand_pattern(pattern[: match.start()] + alt + pattern[match.end() :]))
        return results
    return [pattern]


class ConverterRule(BaseModel):
    """Rule matching a file pathspec to a converter or route name."""

    pathspec: str = Field(description="Gitignore-style path pattern (e.g. '**/*.xlsx' or '**/*.{docx,doc}')")
    converter: str = Field(description="Converter name or route (e.g. 'messy_xls', 'mistral_ocr', 'via_pdf', 'copy')")

    model_config = ConfigDict(extra="allow")

    def matches(self, path: Path) -> bool:
        """Check if the given file path matches this rule's pathspec pattern."""
        import pathspec

        patterns = _expand_pattern(self.pathspec)
        spec = pathspec.PathSpec.from_lines("gitignore", patterns)
        return bool(spec.match_file(str(path)) or spec.match_file(path.name))


class MarkdownizeProfile(BaseModel):
    """Conversion profile containing ordered routing rules for document conversion."""

    name: str = Field(default="custom", description="Profile name")
    rules: list[ConverterRule] = Field(default_factory=list, description="Ordered list of converter routing rules")

    # Backwards-compatibility fields
    ppt_converter: str | None = Field(default=None, description="Legacy PowerPoint converter choice")
    doc_converter: str | None = Field(default=None, description="Legacy Word converter choice")
    excel_converter: str | None = Field(default=None, description="Legacy spreadsheet converter choice")
    pdf_converter: str | None = Field(default=None, description="Legacy PDF converter choice")

    model_config = ConfigDict(extra="allow")

    def model_post_init(self, __context: dict) -> None:
        """Ensure default rules are populated when created with legacy fields or empty rules."""
        if not self.rules:
            # Build rules from legacy converter settings if available
            generated_rules: list[ConverterRule] = []

            # Pre-existing markdown is always copied
            generated_rules.append(ConverterRule(pathspec="**/*.{md,markdown}", converter="copy"))

            if self.excel_converter:
                conv = self.excel_converter
                generated_rules.append(ConverterRule(pathspec="**/*.{xlsx,xls,ods}", converter=conv))

            if self.ppt_converter:
                conv = self.ppt_converter
                generated_rules.append(ConverterRule(pathspec="**/*.{pptx,ppt,odp}", converter=conv))

            if self.doc_converter:
                conv = self.doc_converter
                generated_rules.append(ConverterRule(pathspec="**/*.{docx,doc,odt,rtf}", converter=conv))

            if self.pdf_converter:
                conv = "mistral_ocr" if self.pdf_converter == "mistral" else self.pdf_converter
                generated_rules.append(ConverterRule(pathspec="**/*.pdf", converter=conv))

            # Catch-all default rule
            generated_rules.append(ConverterRule(pathspec="**/*", converter="markitdown"))
            self.rules = generated_rules

    def select_route(self, path: Path) -> str:
        """Evaluate rules in order and return the converter or action route for the given file."""
        suffix = path.suffix.lower()
        if suffix in MD_EXTENSIONS:
            return "copy"

        for rule in self.rules:
            if rule.matches(path):
                return rule.converter

        return "markitdown"

    def fingerprint(self) -> str:
        """Stable cache fingerprint for the profile rules."""
        if self.ppt_converter or self.doc_converter or self.excel_converter or self.pdf_converter:
            return f"{self.ppt_converter}:{self.doc_converter}:{self.excel_converter}:{self.pdf_converter}"
        rule_signatures = [f"{r.pathspec}->{r.converter}" for r in self.rules]
        return ";".join(rule_signatures)
