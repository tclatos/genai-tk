"""Manifest models tracking markdownize conversion results, to avoid reprocessing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MarkdownizeManifestEntry(BaseModel):
    """A single markdown conversion entry in the manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarkdownizeManifest(BaseModel):
    """Manifest for markdown conversion results to avoid reprocessing."""

    entries: dict[str, MarkdownizeManifestEntry] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize manifest to JSON string."""
        return super().model_dump_json(**kwargs)
