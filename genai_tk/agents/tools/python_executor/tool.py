"""LangChain BaseTool wrapper for safe in-process Python execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from genai_tk.agents.tools.python_executor.executor import LocalPythonExecutor
from genai_tk.agents.tools.python_executor.models import PythonExecutorInput


class PythonExecutorTool(BaseTool):
    """LangChain tool for executing Python code in-process via a safe AST interpreter."""

    name: str = "python_interpreter"
    description: str = (
        "Executes Python code in-process using a safe AST interpreter with state persistence across calls. "
        "Captures print output and returns the final evaluated expression or statement result. "
        "Supports standard libraries and authorized packages (e.g., pandas, math, json)."
    )
    args_schema: type[PythonExecutorInput] = PythonExecutorInput
    executor: LocalPythonExecutor = Field(default_factory=LocalPythonExecutor)
    include_logs: bool = Field(
        default=True, description="Whether to include printed logs in the returned string output"
    )

    model_config = {"arbitrary_types_allowed": True}

    def _format_result(self, code_output: Any) -> str:
        if code_output.error:
            err_msg = f"Error: {code_output.error}"
            if code_output.logs:
                return f"Logs:\n{code_output.logs}\n\n{err_msg}"
            return err_msg

        parts = []
        if self.include_logs and code_output.logs:
            parts.append(f"Logs:\n{code_output.logs.rstrip()}")

        if code_output.output is not None:
            parts.append(f"Result:\n{code_output.output}")

        if not parts:
            return "Code executed successfully with no output."

        return "\n\n".join(parts)

    def _run(self, code: str) -> str:
        """Synchronously execute Python code."""
        result = self.executor(code)
        return self._format_result(result)

    async def _arun(self, code: str) -> str:
        """Asynchronously execute Python code in worker thread."""
        return await asyncio.to_thread(self._run, code)


def create_python_executor_tool(
    authorized_imports: list[str] | None = None,
    additional_authorized_imports: list[str] | None = None,
    tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
) -> BaseTool:
    """Factory function to instantiate a configured PythonExecutorTool.

    Args:
        authorized_imports: Additional module names allowed to be imported (e.g. ['pandas', 'numpy']).
        additional_authorized_imports: Alias for authorized_imports.
        tools: LangChain tools or custom functions to expose inside the Python environment.
        timeout_seconds: Max execution time allowed in seconds.
        include_logs: Whether stdout printed logs should be included in the tool output.
        initial_state: Initial variables injected into the execution state.

    Returns:
        Configured PythonExecutorTool instance.
    """
    effective_imports = (authorized_imports or []) + (additional_authorized_imports or [])
    executor = LocalPythonExecutor(
        additional_authorized_imports=effective_imports,
        tools=tools,
        timeout_seconds=timeout_seconds,
        initial_state=initial_state,
    )
    return PythonExecutorTool(executor=executor, include_logs=include_logs)


def create_python_executor_tools(
    authorized_imports: list[str] | None = None,
    additional_authorized_imports: list[str] | None = None,
    tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
) -> list[BaseTool]:
    """Factory returning a list of tools for agent profile integration."""
    return [
        create_python_executor_tool(
            authorized_imports=authorized_imports,
            additional_authorized_imports=additional_authorized_imports,
            tools=tools,
            timeout_seconds=timeout_seconds,
            include_logs=include_logs,
            initial_state=initial_state,
        )
    ]
