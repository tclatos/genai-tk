"""Document to Markdown conversion toolkit."""

from __future__ import annotations

from genai_tk.extra.markdownize.anydoc_converter import AnyDocConverter
from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.extra.markdownize.edgeparse_converter import EdgeParseConverter
from genai_tk.extra.markdownize.excel_converter import MessyExcelConverter
from genai_tk.extra.markdownize.factory import BUILTIN_CONVERTERS, ConverterFactory
from genai_tk.extra.markdownize.lighton_ocr_converter import LightOnOCRConverter
from genai_tk.extra.markdownize.llm_converter import LLMConverter
from genai_tk.extra.markdownize.markitdown_converter import MarkItDownConverter
from genai_tk.extra.markdownize.mistral_ocr_converter import MistralOCRConverter
from genai_tk.extra.markdownize.selector import ConverterRule, MarkdownizeProfile

__all__ = [
    "AnyDocConverter",
    "BUILTIN_CONVERTERS",
    "ConverterFactory",
    "ConverterRule",
    "DocumentConverter",
    "EdgeParseConverter",
    "LLMConverter",
    "LightOnOCRConverter",
    "MarkdownizeProfile",
    "MarkItDownConverter",
    "MessyExcelConverter",
    "MistralOCRConverter",
]
