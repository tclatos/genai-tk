"""LLM-based document to Markdown converter using LangChain LLM factory."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path

from loguru import logger
from pydantic import Field

from genai_tk.core.factories.llm_factory import get_llm
from genai_tk.extra.markdownize.base import DocumentConverter

_LLM_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".html",
    ".htm",
    ".txt",
    ".csv",
    ".json",
    ".md",
}

_TEXT_EXTENSIONS = {".html", ".htm", ".txt", ".csv", ".json", ".md"}

DEFAULT_SYSTEM_PROMPT = """Convert the provided document into clean, high-quality GitHub-Flavored Markdown.
Preserve all headings, tables, bulleted and numbered lists, code blocks, bold/italic formatting, and mathematical expressions.
Output ONLY the resulting Markdown content without conversational preamble, commentary, or markdown code fence wrapping the entire output."""


class LLMConverter(DocumentConverter):
    """Document converter that invokes multimodal or text LLMs via the LangChain LLM factory."""

    llm: str = Field(default="default", description="LLM identifier or tag to resolve via LLM factory")
    prompt_template: str | None = Field(default=None, description="Custom prompt instructions for Markdown extraction")
    max_concurrency: int = Field(default=5, description="Maximum concurrent async LLM invocations")
    temperature: float = Field(default=0.0, description="Model sampling temperature")
    extra_params: dict = Field(default_factory=dict, description="Additional keyword arguments passed to get_llm")

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by LLM converter."""
        return _LLM_SUPPORTED_EXTENSIONS

    def _get_model(self):
        """Instantiate the configured LangChain chat model."""
        return get_llm(llm=self.llm, temperature=self.temperature, **self.extra_params)

    async def convert(self, path: Path) -> str:
        """Convert a document file to Markdown text using the configured LLM."""
        from langchain_core.messages import HumanMessage

        model = self._get_model()
        prompt = self.prompt_template or DEFAULT_SYSTEM_PROMPT
        suffix = path.suffix.lower()

        try:
            if suffix in _TEXT_EXTENSIONS:
                text_content = path.read_text(encoding="utf-8", errors="replace")
                message = HumanMessage(
                    content=f"{prompt}\n\nDocument source: `{path.name}`\n\n```\n{text_content}\n```"
                )
            else:
                raw_bytes = path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(path))
                if suffix == ".pdf":
                    mime_type = "application/pdf"
                mime_type = mime_type or "application/octet-stream"

                b64_content = base64.b64encode(raw_bytes).decode("utf-8")
                data_url = f"data:{mime_type};base64,{b64_content}"
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": f"{prompt}\n\nDocument file: `{path.name}`"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )

            response = await model.ainvoke([message])
            content = response.content
            if isinstance(content, list):
                # Multimodal response parts
                text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
                return "".join(text_parts).strip()
            return str(content).strip()

        except Exception as exc:
            logger.error(f"LLM conversion failed for {path.name} with model '{self.llm}': {exc}")
            raise RuntimeError(
                f"LLM converter failed for {path.name} with model '{self.llm}'. "
                f"Ensure the provider and model support the file format ({suffix}). Error: {exc}"
            ) from exc

    async def batch_convert(self, paths: list[Path]) -> dict[str, str]:
        """Convert a batch of files asynchronously with bounded concurrency."""
        if not paths:
            return {}

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _convert_with_semaphore(path: Path) -> tuple[str, str]:
            async with semaphore:
                text = await self.convert(path)
                return str(path), text

        tasks = [_convert_with_semaphore(p) for p in paths]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, str] = {}
        for path, outcome in zip(paths, outcomes, strict=False):
            if isinstance(outcome, Exception):
                logger.error(f"Batch LLM conversion failed for {path.name}: {outcome}")
                raise outcome
            key, val = outcome
            results[key] = val

        return results
