"""Rich-based middleware for deer-flow agent message tracing.

Provides clear, structured display of agent interactions similar to ReAct agent,
showing user messages, AI responses, tool calls, and tool results in Rich panels.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from genai_tk.utils.markdown import looks_like_markdown

# Patterns emitted by reasoning/thinking models (DeepSeek, Qwen, o1-style wrappers)
# that leak their chain-of-thought into the message content field.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Some OpenRouter model wrappers interleave reasoning keywords inline:
# "analysisXxx...assistantanalysisXxx...assistantcommentary to=functions...assistantfinalActual response"
_INLINE_REASONING_RE = re.compile(
    r"\b(analysis|assistantanalysis|assistantcommentary|assistantcommentary\s+to=\S+[^\n]*)\b.*?(?=\bassistantfinal\b|$)",
    re.DOTALL,
)
_ASSISTANT_FINAL_RE = re.compile(r"assistantfinal(.*)", re.DOTALL)


def _clean_ai_content(content: str) -> str:
    """Strip reasoning/thinking sections from model response content.

    Handles:
    - ``<think>...</think>`` blocks (DeepSeek, Qwen, etc.)
    - Inline ``analysis...assistantfinal<text>`` pattern (some OpenRouter wrappers)
    """
    # Remove <think>...</think> blocks
    content = _THINK_TAG_RE.sub("", content)

    # If content has the "assistantfinal" marker, keep only that part
    final_match = _ASSISTANT_FINAL_RE.search(content)
    if final_match:
        content = final_match.group(1)

    return content.strip()


class DeerFlowRichTraceMiddleware:
    """Middleware that provides Rich-based message tracing for deer-flow agents.

    Displays agent interactions in a structured, readable format with:
    - User messages in blue panels
    - AI responses in royal blue panels with markdown rendering
    - Tool calls in cyan panels with tool name and arguments
    - Tool results in green panels with markdown rendering

    This middleware intercepts the agent's message stream and displays
    messages as they flow through the system, providing transparency
    into the agent's reasoning and actions.
    """

    def __init__(self, console: Console | None = None, max_result_chars: int = 4000) -> None:
        """Initialize the trace middleware.

        Args:
            console: Rich console instance for output. Creates new one if None.
            max_result_chars: Maximum characters to display for tool results
        """
        self._console = console or Console()
        self._max_result_chars = max_result_chars
        self._last_displayed_message_id = None
        self._tool_calls_in_progress: set[str] = set()

    def _print_human_message(self, message: HumanMessage) -> None:
        """Display a human message in a blue panel."""
        content = message.content
        if isinstance(content, list):
            content = "\n".join(str(block) for block in content)

        self._console.print(
            Panel(
                str(content),
                title="[bold blue]User[/bold blue]",
                border_style="blue",
            )
        )

    def _print_ai_message(self, message: AIMessage) -> None:
        """Display an AI message response in a royal blue panel with markdown."""
        content = message.content
        if isinstance(content, list):
            content = "\n".join(str(block) for block in content)

        # Strip reasoning/thinking sections emitted by some models
        content = _clean_ai_content(str(content))

        # Only display if there's actual content (not just tool calls)
        if content and content.strip():
            if looks_like_markdown(content):
                body = Markdown(content)
            else:
                body = content

            self._console.print(
                Panel(
                    body,
                    title="[bold white on royal_blue1] Assistant [/bold white on royal_blue1]",
                    border_style="royal_blue1",
                )
            )

    def _print_tool_call(self, tool_call: dict[str, Any] | Any) -> None:
        """Display a tool call in a cyan panel.

        Args:
            tool_call: Tool call dict or ToolCall object from LangChain
        """
        # Handle both dict and ToolCall object
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "<unknown>")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
        else:
            # ToolCall object from LangChain
            tool_name = getattr(tool_call, "name", "<unknown>")
            tool_args = getattr(tool_call, "args", {})
            tool_id = getattr(tool_call, "id", "")

        # Track that we're displaying this tool call
        if tool_id:
            self._tool_calls_in_progress.add(tool_id)

        # Format arguments nicely with better truncation for long values
        args_text = Text()
        for key, value in tool_args.items():
            args_text.append(f"{key}: ", style="bold cyan")
            # Truncate very long argument values
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:197] + "..."
            args_text.append(f"{value_str}\n", style="white")

        self._console.print(
            Panel(
                args_text if args_text.plain else "[dim]No arguments[/dim]",
                title=f"[bold white on blue]🔧 Tool Call[/bold white on blue] [yellow]{tool_name}[/yellow]",
                border_style="cyan",
            )
        )

    def _print_tool_result(self, message: ToolMessage) -> None:
        """Display a tool result in a green panel.

        Args:
            message: ToolMessage containing the tool execution result
        """
        tool_name = message.name if hasattr(message, "name") else "<unknown>"
        content = message.content

        if isinstance(content, list):
            content = "\n".join(str(block) for block in content)

        # Truncate if too long
        if isinstance(content, str) and len(content) > self._max_result_chars:
            content = content[: self._max_result_chars] + "... [truncated]"

        # Render markdown if applicable
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

        # Clear the in-progress tracking
        tool_id = getattr(message, "tool_call_id", None)
        if tool_id and tool_id in self._tool_calls_in_progress:
            self._tool_calls_in_progress.remove(tool_id)

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process messages before agent execution.

        This is called by deer-flow's middleware system at the start of agent processing.
        """
        # Display the latest user message (last HumanMessage in history).
        # Using the last HumanMessage rather than messages[0] ensures multi-turn
        # chat shows the current query, not the first one from conversation history.
        if "messages" in state and state["messages"]:
            last_human = next(
                (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                None,
            )
            if last_human is not None:
                message_id = id(last_human)
                if message_id != self._last_displayed_message_id:
                    self._print_human_message(last_human)
                    self._last_displayed_message_id = message_id

        return state

    def after_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process messages after model generates a response.

        This is called by deer-flow's middleware system after the LLM generates output.
        """
        if "messages" not in state or not state["messages"]:
            return state

        latest_message = state["messages"][-1]

        if isinstance(latest_message, AIMessage):
            # Check for tool calls
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                # Display tool calls
                for tool_call in latest_message.tool_calls:
                    self._print_tool_call(tool_call)
            else:
                # Display AI response
                self._print_ai_message(latest_message)

        return state

    def after_tools(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process messages after tools execute.

        This is called by deer-flow's middleware system after tool execution.
        """
        if "messages" not in state or not state["messages"]:
            return state

        latest_message = state["messages"][-1]

        if isinstance(latest_message, ToolMessage):
            self._print_tool_result(latest_message)

        return state

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process messages before model invocation.

        Called before the LLM is invoked. Currently a pass-through.
        """
        return state


class SimpleDeerFlowTracer:
    """Simple tracer that logs LLM interactions in a clean format.

    Alternative to the full middleware approach - works as a callback handler
    that can be attached to the LLM to trace its interactions.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the tracer.

        Args:
            console: Rich console for output. Creates new one if None.
        """
        self._console = console or Console()

    def log_interaction(self, node: str, message: Any) -> None:
        """Log a single interaction node and message.

        Args:
            node: The node name (e.g., "model", "tools", "agent")
            message: The message object to log
        """
        if isinstance(message, HumanMessage):
            self._console.print(f"[dim]→ Node: {node}[/dim]")
            self._console.print(
                Panel(
                    str(message.content),
                    title="[bold blue]User[/bold blue]",
                    border_style="blue",
                )
            )
        elif isinstance(message, AIMessage):
            self._console.print(f"[dim]→ Node: {node}[/dim]")

            # Check for tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in message.tool_calls]
                self._console.print(f"[cyan]🔧 Calling tools: {', '.join(tool_names)}[/cyan]")

            # Display content if present
            if message.content:
                content = message.content
                if isinstance(content, list):
                    content = "\n".join(str(block) for block in content)

                if looks_like_markdown(str(content)):
                    body = Markdown(str(content))
                else:
                    body = str(content)

                self._console.print(
                    Panel(
                        body,
                        title="[bold white on royal_blue1] Assistant [/bold white on royal_blue1]",
                        border_style="royal_blue1",
                    )
                )
        elif isinstance(message, ToolMessage):
            tool_name = message.name if hasattr(message, "name") else "unknown"
            self._console.print(f"[green]✓ Tool result: {tool_name}[/green]")

            # Log truncated result
            content = str(message.content)
            if len(content) > 200:
                content = content[:200] + "..."
            logger.debug(f"Tool result preview: {content}")
