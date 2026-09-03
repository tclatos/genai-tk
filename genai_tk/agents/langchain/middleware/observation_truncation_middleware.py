"""Middleware to truncate large tool outputs / observations in agent context.

Long tool outputs (such as massive tabular markdown sections or raw data dumps)
rapidly consume context window tokens across multi-turn agent conversations.
This middleware intercepts tool results and truncates responses exceeding a
configurable size limit, keeping both the head (headers/context) and tail
(footers/totals/conclusions) while inserting a clear truncation notice.

Example YAML configuration::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.observation_truncation_middleware.ObservationTruncationMiddleware
        max_chars: 10000
        head_ratio: 0.8
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from loguru import logger


class ObservationTruncationMiddleware(AgentMiddleware):
    """Intercept tool executions and truncate oversized observations.

    Args:
        max_chars: Maximum character count allowed for a single tool observation (default: 12,000).
        max_lines: Optional maximum line count allowed (default: None).
        head_ratio: Fraction of max_chars allocated to the head of the output (default: 0.8).
        tools: Optional list or set of specific tool names to truncate. When None, applies to all tools.
        excluded_tools: Optional list or set of tool names to never truncate.
    """

    def __init__(
        self,
        max_chars: int = 12_000,
        max_lines: int | None = None,
        head_ratio: float = 0.8,
        tools: list[str] | set[str] | None = None,
        excluded_tools: list[str] | set[str] | None = None,
    ) -> None:
        self.max_chars = max(10, max_chars)
        self.max_lines = max_lines
        self.head_ratio = max(0.1, min(0.9, head_ratio))
        self.tools = frozenset(tools) if tools is not None else None
        self.excluded_tools = frozenset(excluded_tools) if excluded_tools is not None else frozenset()

    def _extract_tool_metadata(self, request: Any) -> tuple[str, Any]:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})
        return tool_name, tool_args

    def _should_truncate(self, tool_name: str) -> bool:
        if tool_name in self.excluded_tools:
            return False
        if self.tools is not None and tool_name not in self.tools:
            return False
        return True

    def _truncate_text(self, text: str, tool_name: str) -> str:
        """Truncate *text* if it exceeds max_chars or max_lines, preserving head and tail."""
        length = len(text)
        lines = text.splitlines(keepends=True)
        line_count = len(lines)

        exceeds_chars = length > self.max_chars
        exceeds_lines = self.max_lines is not None and line_count > self.max_lines

        if not (exceeds_chars or exceeds_lines):
            return text

        head_char_budget = int(self.max_chars * self.head_ratio)
        tail_char_budget = self.max_chars - head_char_budget

        if exceeds_lines and not exceeds_chars:
            assert self.max_lines is not None
            head_lines_count = int(self.max_lines * self.head_ratio)
            tail_lines_count = self.max_lines - head_lines_count
            head_part = "".join(lines[:head_lines_count])
            tail_part = "".join(lines[-tail_lines_count:])
            omitted_lines = line_count - (head_lines_count + tail_lines_count)
            notice = (
                f"\n\n[... Truncated {omitted_lines} lines from '{tool_name}' output. "
                f"Showing first {head_lines_count} and last {tail_lines_count} lines. ...]\n\n"
            )
            logger.info("Truncated tool '{}' observation: {} -> {} lines", tool_name, line_count, self.max_lines)
            return head_part + notice + tail_part

        head_part = text[:head_char_budget]
        tail_part = text[-tail_char_budget:] if tail_char_budget > 0 else ""
        omitted_chars = length - (len(head_part) + len(tail_part))
        notice = (
            f"\n\n[... Truncated {omitted_chars:,} characters from '{tool_name}' observation "
            f"(total was {length:,} chars). Showing head ({len(head_part):,} chars) and tail ({len(tail_part):,} chars). "
            f"Please refine your query or request a specific section if more detail is required. ...]\n\n"
        )
        logger.info("Truncated tool '{}' observation: {} -> {} chars", tool_name, length, self.max_chars)
        return head_part + notice + tail_part

    def _process_result(self, result: Any, tool_name: str) -> Any:
        if isinstance(result, str):
            return self._truncate_text(result, tool_name)
        return result

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Intercept sync tool execution and truncate observation if needed."""
        tool_name, _ = self._extract_tool_metadata(request)
        result = handler(request)
        if self._should_truncate(tool_name):
            return self._process_result(result, tool_name)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """Intercept async tool execution and truncate observation if needed."""
        tool_name, _ = self._extract_tool_metadata(request)
        result = await handler(request)
        if self._should_truncate(tool_name):
            return self._process_result(result, tool_name)
        return result
