"""Custom LangChain agent middleware utilities.

This module defines reusable AgentMiddleware implementations that:
- Log tool calls and results using Rich

These middlewares are intended for CLI-style usage where printing to the
terminal provides a better user experience and simplifies tracing of
agent execution.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from genai_tk.utils.markdown import looks_like_markdown


class RichToolCallMiddleware(AgentMiddleware):
    """Middleware that logs tool calls and their results using Rich."""

    def __init__(self, console: Console | None = None, max_result_chars: int = 4000) -> None:
        self._console = console or Console()
        self._max_result_chars = max_result_chars

    @staticmethod
    def _extract_tool_metadata(request: Any) -> tuple[str, Any]:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})
        return tool_name, tool_args

    def _print_tool_call(self, tool_name: str, tool_args: Any) -> None:
        self._console.print(
            Panel(
                f"[bold cyan]Calling tool[/bold cyan] [yellow]{tool_name}[/yellow]\n\n{tool_args}",
                title="[bold blue]Tool Call[/bold blue]",
                border_style="cyan",
            )
        )

    def _print_tool_result(self, tool_name: str, response: Any) -> None:
        content = getattr(response, "content", str(response))
        if isinstance(content, list):  # e.g. structured content blocks
            content = "\n".join(str(block) for block in content)

        if isinstance(content, str) and len(content) > self._max_result_chars:
            content = content[: self._max_result_chars] + "... [truncated]"

        if looks_like_markdown(str(content)):
            body = Markdown(str(content))
        else:
            body = str(content)

        self._console.print(
            Panel(
                body,
                title=f"[bold green]Tool Result: {tool_name}[/bold green]",
                border_style="green",
            )
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        """Log tool call details before and after execution (sync)."""
        tool_name, tool_args = self._extract_tool_metadata(request)
        self._print_tool_call(tool_name, tool_args)

        response = handler(request)

        self._print_tool_result(tool_name, response)
        return response

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:  # type: ignore[override]
        """Log tool call details before and after execution (async)."""
        tool_name, tool_args = self._extract_tool_metadata(request)
        self._print_tool_call(tool_name, tool_args)

        response = await handler(request)

        self._print_tool_result(tool_name, response)
        return response


def create_rich_agent_middlewares(console: Console | None = None) -> list[AgentMiddleware]:
    """Create the default set of Rich-based middlewares for agents.

    The same console instance is shared between middlewares to ensure
    consistent rendering and avoid duplicated output streams.
    """
    shared_console = console or Console()
    return [RichToolCallMiddleware(console=shared_console)]
