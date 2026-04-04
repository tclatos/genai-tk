"""Shared Rich terminal display helpers for agent CLI commands.

Provides consistent tool-call rendering across all agent backends
(LangChain, DeerFlow, SmolAgents).
"""

from __future__ import annotations

import re

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ASSISTANT_PANEL_TITLE = "[bold white on royal_blue1] Assistant [/bold white on royal_blue1]"
ASSISTANT_BORDER_STYLE = "royal_blue1"


def assistant_panel(text: str, *, markdown: bool = True) -> Panel:
    """Build a Rich Panel for the assistant response."""
    body = Markdown(text) if markdown else text
    padding = (1, 2) if markdown else (0, 0)
    return Panel(body, title=ASSISTANT_PANEL_TITLE, border_style=ASSISTANT_BORDER_STYLE, padding=padding)


# ---------------------------------------------------------------------------
# Tool result summarization
# ---------------------------------------------------------------------------


def summarize_tool_result(tool_name: str, result_text: str, *, max_chars: int = 80) -> str:
    """Produce a short one-line summary of a tool result.

    Smart summaries for known tool types (SQL, planning, file reads);
    generic truncation for everything else.  Used by both the
    RichToolCallMiddleware (langchain) and the deerflow event renderer.
    """
    text = result_text.strip()

    if tool_name == "sql_db_list_tables":
        tables = [t.strip() for t in text.split(",") if t.strip()]
        if tables:
            preview = ", ".join(tables[:5])
            suffix = f", … ({len(tables)} tables)" if len(tables) > 5 else f" ({len(tables)} tables)"
            return preview + suffix

    if tool_name == "sql_db_query_checker":
        if "looks correct" in text.lower() or text.strip().upper().startswith("SELECT"):
            return "query OK"
        return text[:max_chars]

    if tool_name == "sql_db_query":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        n = len(lines)
        if n <= 1:
            return text[:max_chars]
        return f"{n} rows returned"

    if tool_name == "sql_db_schema":
        tables_found = re.findall(r"CREATE TABLE [\"`]?(\w+)[\"`]?", text)
        if tables_found:
            return ", ".join(tables_found)

    if tool_name == "write_todos":
        return "plan updated"

    if tool_name == "tavily_search":
        return f"{len(text)} chars" if len(text) > max_chars else text

    if tool_name == "read_file" and "SKILL.md" in text[:200]:
        return "skill loaded"

    # Sandbox / code execution results
    if tool_name in ("bash", "execute_code", "run_code"):
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "(empty output)"
        if len(lines) == 1:
            return lines[0][:max_chars]
        return f"{len(lines)} lines of output"

    # Generic fallback
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# Tool-call step renderer
# ---------------------------------------------------------------------------


class ToolStepRenderer:
    """Numbered tool-call renderer for CLI output.

    Tracks step numbers and formats tool calls + results consistently.
    Create one per conversation turn to reset numbering.

    Example output::

        🔧 Execution
          1. tavily_search → 2481 chars
          2. bash → 15 lines of output
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._step = 0
        self._header_printed = False

    def _ensure_header(self) -> None:
        if not self._header_printed:
            self._console.print("\n[bold cyan]🔧 Execution[/bold cyan]")
            self._header_printed = True

    def log_tool_call(self, tool_name: str, args: dict | str | None = None) -> None:
        """Log a tool call (before execution)."""
        self._ensure_header()
        self._step += 1
        args_preview = ""
        if args:
            args_str = str(args)[:120].replace("\n", " ")
            args_preview = f" [dim]{args_str}[/dim]"
        self._console.print(f"  [dim]{self._step}.[/dim] [yellow]{tool_name}[/yellow]{args_preview}")

    def log_tool_result(self, tool_name: str, content: str | None, *, is_error: bool = False) -> None:
        """Log a tool result (after execution)."""
        content = content or ""
        if not is_error:
            brief = summarize_tool_result(tool_name, content)
            self._console.print(f"     [dim green]→ {brief}[/dim green]")
        else:
            preview = content[:120].replace("\n", " ")
            self._console.print(f"     [dim red]✗ {preview}[/dim red]")

    def log_combined(self, tool_name: str, content: str | None, *, is_error: bool = False) -> None:
        """Log a tool call + result on one line (compact mode)."""
        self._ensure_header()
        self._step += 1
        content = content or ""
        if is_error:
            preview = content[:80].replace("\n", " ")
            self._console.print(
                f"  [dim]{self._step}.[/dim] [yellow]{tool_name}[/yellow] [dim red]✗ {preview}[/dim red]"
            )
        else:
            brief = summarize_tool_result(tool_name, content)
            self._console.print(f"  [dim]{self._step}.[/dim] [yellow]{tool_name}[/yellow] [dim]→ {brief}[/dim]")
