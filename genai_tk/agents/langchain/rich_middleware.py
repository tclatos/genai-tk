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
from langchain_core.language_models.base import LanguageModelOutput
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from genai_tk.utils.markdown import looks_like_markdown


class SingleToolExecutorMiddleware:
    """Executor for single-tool mode that forces tool execution and chains with display middleware.

    This is not a traditional AgentMiddleware because LangChain's create_agent API doesn't
    expose the right hooks to force tool_choice at the middleware level. Instead, this is
    a pragmatic executor that:
    1. Binds the model with tool_choice to force the LLM to generate tool arguments
    2. Executes the tool
    3. Chains with RichToolCallMiddleware for consistent display

    This approach separates concerns: execution logic here, display in RichToolCallMiddleware.
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        self._executed = False

    async def execute_single_tool(
        self, tool: Any, model: Any, query: str, pre_prompt: str | None = None, tool_wrapper: Callable | None = None
    ) -> str:
        """Execute a single tool with model-generated arguments.

        Args:
            tool: The tool to execute
            model: The LLM model to use for generating tool arguments
            query: User query
            pre_prompt: Optional system prompt
            tool_wrapper: Optional function to wrap tool execution (e.g., RichToolCallMiddleware)

        Returns:
            The tool result as a string
        """
        from langchain_core.messages import HumanMessage

        # Bind the tool with forced tool choice
        bound_model = model.bind_tools([tool], tool_choice=tool.name)

        # Get the model to generate tool arguments
        full_query = f"{pre_prompt}\n\n{query}" if pre_prompt else query
        response = await bound_model.ainvoke([HumanMessage(content=full_query)])

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return "Error: Model did not generate a tool call"

        tool_call = response.tool_calls[0]

        # Create a tool request object that RichToolCallMiddleware can process
        class ToolRequest:
            def __init__(self, name: str, args: dict):
                self.tool_call = {"name": name, "args": args}

        request = ToolRequest(tool.name, tool_call["args"])

        # If we have a tool_wrapper (RichToolCallMiddleware), use it for display
        if tool_wrapper:
            tool_result = await tool_wrapper(request, lambda req: tool.ainvoke(req.tool_call["args"]))
        else:
            # Direct execution without display
            tool_result = await tool.ainvoke(tool_call["args"])

        self._executed = True
        return str(tool_result)


class ToolCallLimitMiddleware(AgentMiddleware):
    """Middleware that limits the number of tool calls and gracefully ends execution.

    This middleware is useful when you want the agent to call a single tool and return
    the raw result without further processing. Designed for non-interactive/direct mode.
    """

    def __init__(self, run_limit: int = 1, thread_limit: int | None = None, exit_behavior: str = "end") -> None:
        """Initialize the middleware with tool call limits.

        Args:
            run_limit: Maximum number of tool calls per turn/run (default: 1)
            thread_limit: Maximum number of tool calls across all turns (default: None for no limit)
            exit_behavior: How to handle limit reached - "end" for graceful stop, "error" for exception
        """
        self._run_limit = run_limit
        self._thread_limit = thread_limit
        self._exit_behavior = exit_behavior
        self._run_count = 0
        self._thread_count = 0

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        """Enforce tool call limit (sync)."""
        self._run_count += 1
        self._thread_count += 1

        if self._run_count > self._run_limit or (self._thread_limit and self._thread_count > self._thread_limit):
            if self._exit_behavior == "error":
                msg = f"Tool call limit exceeded: {self._run_count}/{self._run_limit}"
                raise RuntimeError(msg)
            # For "end" behavior, still execute this call but signal completion
            # The agent framework should handle this gracefully

        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:  # type: ignore[override]
        """Enforce tool call limit (async)."""
        self._run_count += 1
        self._thread_count += 1

        if self._run_count > self._run_limit or (self._thread_limit and self._thread_count > self._thread_limit):
            if self._exit_behavior == "error":
                msg = f"Tool call limit exceeded: {self._run_count}/{self._run_limit}"
                raise RuntimeError(msg)
            # For "end" behavior, still execute this call but signal completion
            # The agent framework should handle this gracefully

        return await handler(request)


class RichToolCallMiddleware(AgentMiddleware):
    """Middleware that traces LLM calls and tool calls to the terminal using Rich."""

    def __init__(self, console: Console | None = None, max_result_chars: int = 4000) -> None:
        self._console = console or Console()
        self._max_result_chars = max_result_chars
        self._call_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_metadata(request: Any) -> tuple[str, Any]:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})
        return tool_name, tool_args

    @staticmethod
    def _model_name(request: Any) -> str:
        model = getattr(request, "model", None)
        if model is None:
            return "<unknown>"
        # Try common attribute names across providers
        for attr in ("model", "model_name", "_model_name", "model_id"):
            val = getattr(model, attr, None)
            if isinstance(val, str) and val:
                return val
        return type(model).__name__

    def _print_tool_call(self, tool_name: str, tool_args: Any) -> None:
        args_str = str(tool_args) if tool_args else "(no args)"
        # Highlight skill reads with a special style
        is_skill_read = tool_name == "read_file" and "SKILL.md" in args_str
        if is_skill_read:
            title = "[bold yellow]📖 Reading Skill[/bold yellow]"
            border = "yellow"
        else:
            title = "[bold blue]⚙ Tool Call[/bold blue]"
            border = "blue"
        self._console.print(
            Panel(
                f"[bold yellow]{tool_name}[/bold yellow]\n\n[dim]{args_str}[/dim]",
                title=title,
                border_style=border,
                padding=(0, 1),
            )
        )

    def _print_tool_result(self, tool_name: str, response: LanguageModelOutput | Any) -> None:
        content = getattr(response, "content", str(response))
        if isinstance(content, list):
            content = "\n".join(str(block) for block in content)
        if isinstance(content, str) and len(content) > self._max_result_chars:
            content = content[: self._max_result_chars] + "… [truncated]"
        body: Any = Markdown(str(content)) if looks_like_markdown(str(content)) else str(content)
        self._console.print(
            Panel(
                body,
                title=f"[bold green]✓ {tool_name}[/bold green]",
                border_style="green",
                padding=(0, 1),
            )
        )

    def _print_llm_call(self, request: Any) -> None:
        model_id = self._model_name(request)
        messages = getattr(request, "messages", [])
        n_msgs = len(messages)
        # Show a brief preview of the last human message if available
        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
            None,
        )
        preview = ""
        if last_human:
            text = getattr(last_human, "content", "")
            if isinstance(text, str):
                preview = text[:120] + ("…" if len(text) > 120 else "")
        tools = getattr(request, "tools", [])
        tools_str = f"  [dim]{len(tools)} tool(s) available[/dim]" if tools else ""
        body = f"[bold cyan]{model_id}[/bold cyan]  [dim]({n_msgs} messages){tools_str}[/dim]"
        if preview:
            body += f"\n[dim italic]{preview}[/dim italic]"

        # Extract skill metadata from system message (injected by SkillsMiddleware)
        sys_msg = getattr(request, "system_message", None)
        sys_text = ""
        if sys_msg:
            content = getattr(sys_msg, "content", None) or sys_msg
            if isinstance(content, list):
                sys_text = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block) for block in content
                )
            elif isinstance(content, str):
                sys_text = content
            else:
                sys_text = str(content)
        if "**Available Skills:**" in sys_text:
            import re

            skills_block = re.search(r"\*\*Available Skills:\*\*\s*(.*?)(?:\*\*How to Use Skills)", sys_text, re.DOTALL)
            if skills_block:
                # Clean up: extract just the skill names from markdown list
                raw = skills_block.group(1).strip()
                names = [m.group(1) for m in re.finditer(r"\*\*([^*]+)\*\*:", raw)]
                if names:
                    body += f"\n[bold]Skills:[/bold] [dim]{', '.join(names)}[/dim]"

        self._call_count += 1
        self._console.print(
            Panel(
                body,
                title=f"[bold magenta]🧠 LLM Call #{self._call_count}[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        tool_name, tool_args = self._extract_tool_metadata(request)
        self._print_tool_call(tool_name, tool_args)
        response = handler(request)
        self._print_tool_result(tool_name, response)
        return response

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:  # type: ignore[override]
        tool_name, tool_args = self._extract_tool_metadata(request)
        self._print_tool_call(tool_name, tool_args)
        response = await handler(request)
        self._print_tool_result(tool_name, response)
        return response

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:  # type: ignore[override]
        """Trace each LLM invocation before executing it."""
        self._print_llm_call(request)
        return await handler(request)


def create_rich_agent_middlewares(
    console: Console | None = None, single_tool_mode: bool = False
) -> list[AgentMiddleware]:
    """Create the default set of Rich-based middlewares for agents.

    The same console instance is shared between middlewares to ensure
    consistent rendering and avoid duplicated output streams.

    Note: In single_tool_mode, the actual tool forcing happens via
    SingleToolExecutorMiddleware.execute_single_tool() which is called directly,
    not through the middleware chain. This is pragmatic since LangChain's create_agent
    doesn't expose hooks for forcing tool_choice at the middleware level.

    Args:
        console: Optional Rich console instance to use
        single_tool_mode: Currently unused; kept for API compatibility
    """
    shared_console = console or Console()
    return [RichToolCallMiddleware(console=shared_console)]
