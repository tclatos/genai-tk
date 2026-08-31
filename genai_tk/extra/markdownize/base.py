"""Abstract base class for document to Markdown converters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class DocumentConverter(BaseModel, ABC):
    """Base class for all document to Markdown converters."""

    name: str = Field(default="", description="Converter identifier name")
    batch_enabled: bool = Field(default=False, description="Whether native batch mode is enabled")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return lower-cased file extensions supported by this converter."""
        ...

    @abstractmethod
    async def convert(self, path: Path) -> str:
        """Convert a single document file to Markdown text."""
        ...

    async def batch_convert(self, paths: list[Path]) -> dict[str, str]:
        """Convert a list of files to Markdown, returning path string to Markdown text mapping.

        Args:
            paths: List of file paths to convert.

        Returns:
            Dictionary mapping str(path) to extracted Markdown text.
        """
        if not paths:
            return {}

        results: dict[str, str] = {}
        tasks = [self.convert(p) for p in paths]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for path, outcome in zip(paths, outcomes, strict=False):
            if isinstance(outcome, Exception):
                logger.error(f"Failed to convert {path.name} with {self.__class__.__name__}: {outcome}")
                raise outcome
            results[str(path)] = outcome

        return results
