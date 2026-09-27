"""Unit tests for SandboxBackendFactory and BasePythonExecutor protocol."""

from __future__ import annotations

import pytest

from genai_tk.agents.sandbox.factory import SandboxBackendFactory
from genai_tk.agents.sandbox.manager import DockerSandboxManager, SandboxManager
from genai_tk.agents.tools.python_executor import (
    BasePythonExecutor,
    DockerPythonExecutor,
    LocalPythonExecutor,
    SandboxedPythonExecutor,
)


@pytest.mark.unit
def test_base_python_executor_protocol():
    """Verify LocalPythonExecutor and SandboxedPythonExecutor implement BasePythonExecutor."""
    local_exec = LocalPythonExecutor()
    assert isinstance(local_exec, BasePythonExecutor)

    sandboxed_exec = SandboxedPythonExecutor()
    assert isinstance(sandboxed_exec, BasePythonExecutor)
    assert DockerPythonExecutor is SandboxedPythonExecutor


@pytest.mark.unit
def test_sandbox_backend_factory_builtins():
    """Verify built-in sandbox backend mappings in SandboxBackendFactory."""
    for name in ("docker", "aio_sandbox", "local", "filesystem"):
        path = SandboxBackendFactory.get_backend_class_path(name)
        assert path is not None
        assert isinstance(path, str)


class MockCustomBackend:
    """Mock backend defined at module scope for import resolution."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = "mock-123"

    async def aexecute(self, command: str, timeout: int = 30):
        pass


@pytest.mark.unit
def test_sandbox_backend_factory_register_and_create():
    """Verify custom sandbox backend registration and instantiation."""
    qualified = f"{MockCustomBackend.__module__}.{MockCustomBackend.__name__}"
    SandboxBackendFactory.register("mock_test_backend", qualified)
    resolved = SandboxBackendFactory.get_backend_class_path("mock_test_backend")
    assert "MockCustomBackend" in resolved

    instance = SandboxBackendFactory.create("mock_test_backend", token="secret_token")
    assert instance.id == "mock-123"
    assert instance.kwargs.get("token") == "secret_token"


@pytest.mark.unit
def test_sandbox_manager_singleton():
    """Verify SandboxManager singleton and backward compatibility alias."""
    mgr = SandboxManager.singleton()
    assert mgr is not None
    assert DockerSandboxManager is SandboxManager
