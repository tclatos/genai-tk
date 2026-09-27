"""Base protocol and abstract definition for Python code executors."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genai_tk.agents.tools.python_executor.models import CodeOutput


@runtime_checkable
class BasePythonExecutor(Protocol):
    """Protocol defining the interface for stateful Python code executors."""

    def __call__(self, code: str) -> CodeOutput:
        """Synchronously execute Python code and return structured output."""
        ...

    async def aexecute_code(self, code: str) -> CodeOutput:
        """Asynchronously execute Python code and return structured output."""
        ...

    def reset(self) -> None:
        """Reset internal execution state and namespace."""
        ...

    def send_tools(self, tools: dict[str, Any] | list[Any]) -> None:
        """Register or update tools available to the executor."""
        ...

    def send_variables(self, variables: dict[str, Any]) -> None:
        """Inject variables into the executor's shared state."""
        ...
