"""LangChain BaseTool wrapper for Python code execution via Docker sandbox or safe AST."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import Field

from genai_tk.agents.sandbox.manager import DockerSandboxManager
from genai_tk.agents.tools.python_executor.docker_executor import DockerPythonExecutor
from genai_tk.agents.tools.python_executor.executor import LocalPythonExecutor
from genai_tk.agents.tools.python_executor.models import PythonExecutorInput

COMMON_OPTIONAL_PACKAGES: list[str] = ["numpy", "pandas", "scipy", "sympy"]


class PythonExecutorTool(BaseTool):
    """LangChain tool for executing Python code in a Docker sandbox or safe AST interpreter."""

    name: str = "python_interpreter"
    description: str = (
        "Executes Python code in a stateful sandbox with variable persistence across calls. "
        "Captures print output and returns the final evaluated expression or statement result. "
        "Supports standard libraries and data science packages (e.g., numpy, pandas, scipy, math, json)."
    )
    args_schema: type[PythonExecutorInput] = PythonExecutorInput
    executor: Any = Field(default_factory=LocalPythonExecutor)
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

        if code_output.is_final_answer:
            parts.append(f"FINAL ANSWER:\n{code_output.output}")
        elif code_output.output is not None:
            parts.append(f"Result:\n{code_output.output}")

        if not parts:
            return "Code executed successfully with no output."

        return "\n\n".join(parts)

    def _run(self, code: str) -> str:
        """Synchronously execute Python code."""
        result = self.executor(code)
        return self._format_result(result)

    async def _arun(self, code: str) -> str:
        """Asynchronously execute Python code."""
        if hasattr(self.executor, "aexecute_code"):
            result = await self.executor.aexecute_code(code)
        else:
            result = await asyncio.to_thread(self._run, code)
        return self._format_result(result)


def create_python_executor_tool(
    authorized_imports: list[str] | None = None,
    additional_authorized_imports: list[str] | None = None,
    tools: dict[str, Any] | list[Any] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
    executor_type: str = "auto",
    backend: Any = None,
) -> BaseTool:
    """Factory function to instantiate a configured PythonExecutorTool.

    Args:
        authorized_imports: Additional module names allowed to be imported (e.g. ['pandas', 'numpy']).
        additional_authorized_imports: Alias for authorized_imports.
        tools: Tools, tool factories, qualified names, or custom functions to expose inside Python.
        timeout_seconds: Max execution time allowed in seconds.
        include_logs: Whether stdout printed logs should be included in the tool output.
        initial_state: Initial variables injected into the execution state.
        executor_type: 'docker', 'local', or 'auto' (prefers Docker if available, falls back to local).
        backend: Optional existing AioSandboxBackend to bind to.

    Returns:
        Configured PythonExecutorTool instance.
    """
    discovered = [pkg for pkg in COMMON_OPTIONAL_PACKAGES if find_spec(pkg) is not None]
    effective_imports = list(set((authorized_imports or []) + (additional_authorized_imports or []) + discovered))

    use_docker = False
    if executor_type in ("docker", "sandbox"):
        use_docker = True
    elif executor_type == "auto":
        use_docker = backend is not None or DockerSandboxManager.is_docker_available()

    if use_docker:
        try:
            logger.info("Instantiating DockerPythonExecutor in Docker sandbox...")
            executor = DockerPythonExecutor(
                backend=backend,
                additional_authorized_imports=effective_imports,
                tools=tools,
                timeout_seconds=timeout_seconds,
                initial_state=initial_state,
            )
            return PythonExecutorTool(executor=executor, include_logs=include_logs)
        except Exception as exc:
            if executor_type in ("docker", "sandbox"):
                raise RuntimeError(f"Failed to start Docker sandbox Python executor: {exc}") from exc
            logger.warning(f"Docker sandbox executor initialization failed, falling back to local: {exc}")

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
    tools: dict[str, Any] | list[Any] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
    llm: Any = "default",
    executor_type: str = "auto",
    backend: Any = None,
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
            executor_type=executor_type,
            backend=backend,
        )
    ]
