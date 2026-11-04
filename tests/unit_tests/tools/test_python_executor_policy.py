"""Trust selection regressions at the public Python tool factory."""

from __future__ import annotations

import pytest

from genai_tk.agents.tools.python_executor.docker_executor import DockerPythonExecutor
from genai_tk.agents.tools.python_executor.tool import PythonExecutorTool, create_python_executor_tool


class UnavailableSandbox:
    """External sandbox transport that cannot start."""

    async def start(self) -> None:
        raise ConnectionError("sandbox unavailable")

    async def aexecute(self, command: str, *, timeout: int | None = None) -> None:
        raise ConnectionError("sandbox unavailable")

    async def stop(self) -> None:
        return None


@pytest.mark.unit
def test_direct_tool_defaults_to_isolation() -> None:
    assert isinstance(PythonExecutorTool().executor, DockerPythonExecutor)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["docker", "sandbox", "auto"])
async def test_factory_failure_never_runs_host_tool(mode: str) -> None:
    calls: list[int] = []

    def host_action(value: int) -> int:
        calls.append(value)
        return value

    tool = create_python_executor_tool(
        executor_type=mode, backend=UnavailableSandbox(), tools={"host_action": host_action}
    )
    result = await tool.ainvoke({"code": "host_action(42)"})
    assert "Error:" in result and calls == []


@pytest.mark.unit
def test_unknown_mode_cannot_select_local_execution() -> None:
    with pytest.raises(ValueError, match="Unknown Python executor"):
        create_python_executor_tool(executor_type="doker")


@pytest.mark.unit
def test_installed_packages_are_not_implicitly_authorized_locally() -> None:
    tool = create_python_executor_tool(executor_type="local")
    assert "not allowed" in tool.invoke({"code": "import pandas"})


@pytest.mark.unit
def test_local_mode_rejects_conflicting_backend() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        create_python_executor_tool(executor_type="local", backend=UnavailableSandbox())
