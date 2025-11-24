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

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        """Log tool call details before and after execution.

        The request object is a ToolRequest coming from LangChain. We avoid
        importing the concrete type here to keep dependencies light while
        still providing useful runtime logging.
        """
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})

        self._console.print(
            Panel(
                f"[bold cyan]Calling tool[/bold cyan] [yellow]{tool_name}[/yellow]\n\n{tool_args}",
                title="[bold blue]Tool Call[/bold blue]",
                border_style="cyan",
            )
        )

        response = handler(request)

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

        return response

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:  # type: ignore[override]
        """Async version of wrap_tool_call for use with ainvoke/astream.

        Mirrors the sync implementation but awaits the handler.
        """
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})

        self._console.print(
            Panel(
                f"[bold cyan]Calling tool[/bold cyan] [yellow]{tool_name}[/yellow]\n\n{tool_args}",
                title="[bold blue]Tool Call[/bold blue]",
                border_style="cyan",
            )
        )

        response = await handler(request)

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

        return response


def create_rich_agent_middlewares(console: Console | None = None) -> list[AgentMiddleware]:
    """Create the default set of Rich-based middlewares for agents.

    The same console instance is shared between middlewares to ensure
    consistent rendering and avoid duplicated output streams.
    """
    shared_console = console or Console()
    return [RichToolCallMiddleware(console=shared_console)]
