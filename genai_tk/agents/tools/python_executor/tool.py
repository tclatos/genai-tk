"""LangChain tools for isolated Python execution or explicit trusted local use."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import Field

from genai_tk.agents.tools.python_executor.docker_executor import DockerPythonExecutor
from genai_tk.agents.tools.python_executor.executor import LocalPythonExecutor
from genai_tk.agents.tools.python_executor.models import CodeOutput, PythonExecutorInput

ExecutorType = Literal["docker", "sandbox", "auto", "local"]


class PythonExecutorTool(BaseTool):
    """Execute Python in a dedicated sandbox by default."""

    name: str = "python_interpreter"
    description: str = (
        "Executes Python code in a stateful sandbox with variable persistence across calls. "
        "Captures print output and returns the final evaluated expression or statement result. "
        "Supports standard libraries and data science packages (e.g., numpy, pandas, scipy, math, json)."
    )
    args_schema: type[PythonExecutorInput] = PythonExecutorInput
    executor: DockerPythonExecutor | LocalPythonExecutor = Field(default_factory=DockerPythonExecutor)
    include_logs: bool = Field(
        default=True, description="Whether to include printed logs in the returned string output"
    )

    model_config = {"arbitrary_types_allowed": True}

    def _format_result(self, code_output: CodeOutput) -> str:
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
        result = await self.executor.aexecute_code(code)
        return self._format_result(result)


def create_python_executor_tool(
    authorized_imports: list[str] | None = None,
    additional_authorized_imports: list[str] | None = None,
    tools: dict[str, Any] | list[Any] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
    executor_type: ExecutorType = "docker",
    backend: Any = None,
) -> BaseTool:
    """Factory function to instantiate a configured PythonExecutorTool.

    Args:
        authorized_imports: Explicit additional imports for the trusted local interpreter.
        additional_authorized_imports: Alias for authorized_imports.
        tools: Tool instances or custom functions to expose inside Python. Specify factories with qualified
            names or configuration objects; plain callables are registered without being invoked.
        timeout_seconds: Sandbox wall deadline; local execution offers only a cooperative wait limit.
        include_logs: Whether stdout printed logs should be included in the tool output.
        initial_state: Initial variables injected into the execution state.
        executor_type: 'docker' (default), 'sandbox', or 'auto' requires isolation. 'local' opts into trusted host execution.
        backend: Optional caller-managed sandbox. Its users share a single trust boundary.

    Returns:
        Configured PythonExecutorTool instance.
    """
    if executor_type not in ("docker", "sandbox", "auto", "local"):
        raise ValueError(f"Unknown Python executor type: {executor_type!r}")
    effective_imports = sorted(set((authorized_imports or []) + (additional_authorized_imports or [])))
    if executor_type == "local":
        if backend is not None:
            raise ValueError("A sandbox backend cannot be combined with trusted local execution")
        return PythonExecutorTool(
            executor=LocalPythonExecutor(
                additional_authorized_imports=effective_imports,
                tools=tools,
                timeout_seconds=timeout_seconds,
                initial_state=initial_state,
            ),
            include_logs=include_logs,
            description="Executes trusted Python code on the host. This local interpreter is not a security boundary.",
        )
    return PythonExecutorTool(
        executor=DockerPythonExecutor(
            backend=backend,
            additional_authorized_imports=effective_imports,
            tools=tools,
            timeout_seconds=timeout_seconds,
            initial_state=initial_state,
        ),
        include_logs=include_logs,
    )


def create_python_executor_tools(
    authorized_imports: list[str] | None = None,
    additional_authorized_imports: list[str] | None = None,
    tools: dict[str, Any] | list[Any] | None = None,
    timeout_seconds: int | None = 30,
    include_logs: bool = True,
    initial_state: dict[str, Any] | None = None,
    llm: Any = "default",
    executor_type: ExecutorType = "docker",
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
