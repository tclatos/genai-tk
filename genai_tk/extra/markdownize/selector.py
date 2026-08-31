"""Pathspec-driven converter selection and routing rules."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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

    model_config = ConfigDict(extra="allow")

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
        rule_signatures = [f"{r.pathspec}->{r.converter}" for r in self.rules]
        return ";".join(rule_signatures)
