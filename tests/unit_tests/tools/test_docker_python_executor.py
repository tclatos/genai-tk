"""Regression tests for fail-closed execution and the sandbox tool relay."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import BaseTool, tool

from genai_tk.agents.tools.python_executor.docker_executor import DockerPythonExecutor, WorkerReply
from genai_tk.agents.tools.python_executor.tool import create_python_executor_tool
from genai_tk.agents.tools.tool_specs import FactoryToolSpec, FunctionToolSpec, UnifiedToolSpec


class UnavailableBackend:
    """External transport boundary unavailable before code execution."""

    async def aexecute(self, command: str, *, timeout: int | None = None) -> None:
        raise ConnectionError("Sandbox transport is unavailable.")


class FailingShutdownBackend(UnavailableBackend):
    """External lifecycle boundary that fails its first termination request."""

    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise ConnectionError("Container service unavailable.")


def create_test_tools(multiplier: int = 2) -> list[BaseTool]:
    """Provide a real qualified factory for configuration compatibility tests."""

    @tool
    def scale(value: int) -> int:
        """Scale a number by the configured factor."""
        return value * multiplier

    return [scale]


@pytest.mark.unit
def test_host_fallback_is_rejected() -> None:
    with pytest.raises(ValueError, match="Host fallback is disabled"):
        DockerPythonExecutor(fallback_to_local=True)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["docker", "sandbox", "auto"])
@pytest.mark.parametrize("composite", [False, True])
def test_sandbox_modes_reject_host_shell_backend(tmp_path: Path, mode: str, composite: bool) -> None:
    local_shell = pytest.importorskip("deepagents.backends.local_shell")
    backend = local_shell.LocalShellBackend(root_dir=tmp_path)
    if composite:
        from deepagents.backends.composite import CompositeBackend

        backend = CompositeBackend(default=backend, routes={})
    with pytest.raises(ValueError, match="Host shell backends cannot provide isolation"):
        create_python_executor_tool(executor_type=mode, backend=backend)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_backend_keeps_shared_server_available() -> None:
    pytest.importorskip("opensandbox")
    executor = DockerPythonExecutor()
    with patch("httpx.AsyncClient") as http_client, patch("opensandbox.Sandbox.create", new_callable=AsyncMock):
        http_client.return_value.__aenter__.return_value.get = AsyncMock()
        backend = await executor._get_backend()
    assert backend.stop_server_on_close is False
    await executor.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("spec_kind", ["qualified", "dict", "factory", "function", "unified"])
async def test_configured_tool_factories_are_supported(spec_kind: str) -> None:
    target = f"{__name__}.create_test_tools"
    specs = {
        "qualified": target,
        "dict": {"factory": target, "multiplier": 2},
        "factory": FactoryToolSpec(factory=target, extra_params={"multiplier": 2}),
        "function": FunctionToolSpec(function=target),
        "unified": UnifiedToolSpec(target=target, extra_params={"multiplier": 2}),
    }
    executor = DockerPythonExecutor(tools=[specs[spec_kind]])
    result = await executor._invoke_tool(WorkerReply(kind="tool_call", tool_name="scale", kwargs={"value": 21}))
    assert result == {"result": 42}


@pytest.mark.unit
def test_plain_callable_is_not_eagerly_invoked_as_factory() -> None:
    calls: list[int] = []

    def action() -> int:
        calls.append(1)
        return 42

    DockerPythonExecutor(tools=[action])
    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transport_failure_never_executes_on_host() -> None:
    calls: list[str] = []
    executor = DockerPythonExecutor(backend=UnavailableBackend(), tools={"record": lambda: calls.append("executed")})
    await executor.aexecute_code("record()")
    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transport_failure_invalidates_executor() -> None:
    executor = DockerPythonExecutor(backend=UnavailableBackend())
    await executor.aexecute_code("40 + 2")
    result = await executor.aexecute_code("40 + 2")
    assert "closed or invalidated" in result.error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_revoked_tool_alias_is_denied_at_host() -> None:
    executor = DockerPythonExecutor(tools={"old": lambda: "secret"})
    executor.send_tools({})
    result = await executor._invoke_tool(WorkerReply(kind="tool_call", tool_name="old"))
    assert result == {"error": "Tool is not authorized in this session."}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replacing_tools_revokes_additional_functions() -> None:
    executor = DockerPythonExecutor(additional_functions={"old": lambda: "secret"})
    executor.send_tools({})
    result = await executor._invoke_tool(WorkerReply(kind="tool_call", tool_name="old"))
    assert result == {"error": "Tool is not authorized in this session."}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_failure_is_returned_to_caller() -> None:
    executor = DockerPythonExecutor()
    executor.backend = FailingShutdownBackend()
    result = await executor.aexecute_code("40 + 2")
    assert "termination could not be confirmed" in result.error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_can_be_retried_after_failure() -> None:
    backend = FailingShutdownBackend()
    executor = DockerPythonExecutor()
    executor.backend = backend
    with pytest.raises(RuntimeError, match="termination could not be confirmed"):
        await executor.aclose()
    await executor.aclose()
    assert backend.stop_calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_reset_invalidates_executor() -> None:
    executor = DockerPythonExecutor(backend=UnavailableBackend())
    executor._worker_ready = True
    with pytest.raises(RuntimeError, match="reset failed"):
        await executor._areset()
    with pytest.raises(RuntimeError, match="closed or invalidated"):
        executor.send_variables({"value": 42})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_is_invoked_once() -> None:
    calls: list[int] = []

    def failing_tool() -> str:
        calls.append(1)
        raise RuntimeError("temporary failure")

    executor = DockerPythonExecutor(tools={"action": failing_tool})
    await executor._invoke_tool(WorkerReply(kind="tool_call", tool_name="action"))
    assert calls == [1]


@pytest.mark.unit
def test_initial_state_requires_json_values() -> None:
    with pytest.raises(ValueError):
        DockerPythonExecutor(initial_state={"callback": lambda: None})


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_arithmetic_and_numpy() -> None:
    async with DockerPythonExecutor() as executor:
        result = await executor.aexecute_code("import numpy as np\nint(np.sqrt(100) * 2)")
        assert result.output == 20


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_state_persistence() -> None:
    async with DockerPythonExecutor() as executor:
        await executor.aexecute_code("value = 42")
        result = await executor.aexecute_code("value * 2")
        assert result.output == 84


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_final_answer() -> None:
    async with DockerPythonExecutor() as executor:
        result = await executor.aexecute_code("final_answer({'answer': 42})")
        assert result.is_final_answer


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_host_tool_relay() -> None:
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with DockerPythonExecutor(tools=[add_numbers]) as executor:
        result = await executor.aexecute_code("add_numbers(a=19, b=23)")
        assert result.output == 42


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_default_executors_do_not_share_container() -> None:
    async with DockerPythonExecutor() as first, DockerPythonExecutor() as second:
        await first.aexecute_code("secret_value = 42")
        result = await second.aexecute_code("'secret_value' in globals()")
        assert result.output is False
