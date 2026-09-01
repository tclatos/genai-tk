"""Middleware that deduplicates and caches identical idempotent tool calls within an agent session.

Agents (especially deep/reasoning agents) frequently repeat identical idempotent
read calls (e.g. ``get_document_toc``, ``get_folder_toc``, ``list_documents``)
after reading intermediate sections, even though the full table of contents is
already present earlier in the conversation history.

This middleware intercepts tool requests matching a configured tool list:
- In ``mode="stub"`` (default): Returns a lightweight notice informing the agent
  that the tool output was already retrieved earlier in the session, avoiding
  re-injecting thousands of duplicate TOC tokens into the LLM context.
- In ``mode="cache"``: Returns the verbatim cached response from the previous
  call without re-executing the underlying tool function.

Example YAML configuration::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.deduplicate_middleware.DeduplicateToolCallsMiddleware
        tools: ["get_document_toc", "get_folder_toc", "list_documents"]
        mode: "stub"
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware
from loguru import logger

_DEFAULT_DEDUP_TOOLS = frozenset({"get_document_toc", "get_folder_toc", "list_documents"})


class DeduplicateToolCallsMiddleware(AgentMiddleware):
    """Intercept and deduplicate repeated idempotent tool calls within an agent session.

    Args:
        tools: List or set of tool names to deduplicate. If None, defaults to
            ``{"get_document_toc", "get_folder_toc", "list_documents"}``.
        mode: How to handle duplicate calls:
            - ``"stub"`` (default): return a concise notice directing the agent
              to refer to earlier output in the conversation history.
            - ``"cache"``: return the exact cached result from the earlier call.
        custom_stub: Optional custom message template for stub mode. May contain
            ``{tool_name}`` and ``{args}``.
    """

    def __init__(
        self,
        tools: list[str] | set[str] | None = None,
        mode: Literal["stub", "cache"] = "stub",
        custom_stub: str | None = None,
    ) -> None:
        self.target_tools = frozenset(tools) if tools is not None else _DEFAULT_DEDUP_TOOLS
        self.mode = mode
        self.custom_stub = custom_stub
        self._cache: dict[str, Any] = {}
        self._counts: dict[str, int] = {}

    def _extract_tool_metadata(self, request: Any) -> tuple[str, Any]:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})
        return tool_name, tool_args

    def _make_key(self, tool_name: str, tool_args: Any) -> str:
        if isinstance(tool_args, dict):
            try:
                args_str = json.dumps(tool_args, sort_keys=True)
            except Exception:
                args_str = str(sorted(tool_args.items()))
        else:
            args_str = str(tool_args)
        return f"{tool_name}:{args_str}"

    def _format_stub(self, tool_name: str, tool_args: Any) -> str:
        if self.custom_stub:
            return self.custom_stub.format(tool_name=tool_name, args=tool_args)
        return (
            f"[Notice: Duplicate call to '{tool_name}' with arguments {tool_args}. "
            f"The full result is already present in your conversation history above. "
            f"Please refer to the earlier '{tool_name}' output to select section IDs or continue.]"
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Intercept sync tool execution and deduplicate if previously called."""
        tool_name, tool_args = self._extract_tool_metadata(request)

        if tool_name not in self.target_tools:
            return handler(request)

        key = self._make_key(tool_name, tool_args)
        if key in self._cache:
            count = self._counts.get(key, 1) + 1
            self._counts[key] = count
            logger.info(
                "[DeduplicateToolCalls] Intercepted duplicate call #{} to '{}' (mode={})",
                count,
                tool_name,
                self.mode,
            )
            if self.mode == "stub":
                return self._format_stub(tool_name, tool_args)
            return self._cache[key]

        # First time invocation: execute tool and cache response
        response = handler(request)
        self._cache[key] = response
        self._counts[key] = 1
        return response

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """Intercept async tool execution and deduplicate if previously called."""
        tool_name, tool_args = self._extract_tool_metadata(request)

        if tool_name not in self.target_tools:
            return await handler(request)

        key = self._make_key(tool_name, tool_args)
        if key in self._cache:
            count = self._counts.get(key, 1) + 1
            self._counts[key] = count
            logger.info(
                "[DeduplicateToolCalls] Intercepted duplicate call #{} to '{}' (mode={})",
                count,
                tool_name,
                self.mode,
            )
            if self.mode == "stub":
                return self._format_stub(tool_name, tool_args)
            return self._cache[key]

        # First time invocation: execute tool and cache response
        response = await handler(request)
        self._cache[key] = response
        self._counts[key] = 1
        return response

    def clear(self) -> None:
        """Clear the cached responses and invocation counts."""
        self._cache.clear()
        self._counts.clear()
