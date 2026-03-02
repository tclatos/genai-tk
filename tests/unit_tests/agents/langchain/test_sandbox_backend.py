"""Unit tests for AioSandboxBackend — all mocked, no Docker required."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.agents.langchain.sandbox_backend import (
    AioSandboxBackend,
    AioSandboxBackendConfig,
    SandboxToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shell_response(output: str = "", exit_code: int = 0) -> MagicMock:
    """Build a mock ResponseShellCommandResult."""
    data = MagicMock()
    data.output = output
    data.exit_code = exit_code
    resp = MagicMock()
    resp.data = data
    return resp


def _make_file_read_response(content: str) -> MagicMock:
    data = MagicMock()
    data.content = content
    resp = MagicMock()
    resp.data = data
    return resp


def _make_file_list_response(files: list[tuple[str, int | None]]) -> MagicMock:
    """files: list of (path, size) tuples."""
    file_infos = []
    for path, size in files:
        f = MagicMock()
        f.path = path
        f.size = size
        file_infos.append(f)
    data = MagicMock()
    data.files = file_infos
    resp = MagicMock()
    resp.data = data
    return resp


def _make_file_replace_response(replaced_count: int) -> MagicMock:
    data = MagicMock()
    data.replaced_count = replaced_count
    resp = MagicMock()
    resp.data = data
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> AioSandboxBackend:
    return AioSandboxBackend(config=AioSandboxBackendConfig(startup_timeout=1.0))


@pytest.fixture
def started_backend(backend: AioSandboxBackend) -> AioSandboxBackend:
    """Backend with a mock HTTP client already connected."""
    mock_client = MagicMock()
    mock_client.shell = MagicMock()
    mock_client.shell.exec_command = AsyncMock()
    mock_client.file = MagicMock()
    mock_client.file.read_file = AsyncMock()
    mock_client.file.write_file = AsyncMock()
    mock_client.file.list_path = AsyncMock()
    mock_client.file.replace_in_file = AsyncMock()
    backend._client = mock_client
    return backend


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_list_tools(backend: AioSandboxBackend) -> None:
    assert set(backend.list_tools()) == {"bash", "ls", "read_file", "write_file", "str_replace"}


@pytest.mark.asyncio
async def test_execute_tool_not_started(backend: AioSandboxBackend) -> None:
    with pytest.raises(RuntimeError, match="not started"):
        await backend.execute_tool("bash", {"command": "echo hi"})


@pytest.mark.asyncio
async def test_execute_unknown_tool(started_backend: AioSandboxBackend) -> None:
    with pytest.raises(ValueError, match="Unsupported tool"):
        await started_backend.execute_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_success(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("hello\n", 0))

    result = await started_backend.execute_tool("bash", {"command": "echo hello"})

    assert result.success
    assert result.output == "hello\n"
    assert result.tool_name == "bash"
    started_backend._client.shell.exec_command.assert_awaited_once_with(command="echo hello")


@pytest.mark.asyncio
async def test_bash_failure(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("", 1))

    result = await started_backend.execute_tool("bash", {"command": "bogus"})

    assert not result.success
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ls_uses_path(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.list_path = AsyncMock(return_value=_make_file_list_response([("/tmp/file.txt", 42)]))

    result = await started_backend.execute_tool("ls", {"path": "/tmp"})

    assert result.success
    assert "/tmp/file.txt" in result.output
    started_backend._client.file.list_path.assert_awaited_once_with(path="/tmp", include_size=True, show_hidden=True)


@pytest.mark.asyncio
async def test_ls_defaults_to_work_dir(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.list_path = AsyncMock(return_value=_make_file_list_response([]))

    result = await started_backend.execute_tool("ls", {})

    assert result.success
    started_backend._client.file.list_path.assert_awaited_once_with(
        path="/home/user", include_size=True, show_hidden=True
    )


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("file content"))

    result = await started_backend.execute_tool("read_file", {"path": "/tmp/test.txt"})

    assert result.success
    assert result.output == "file content"
    started_backend._client.file.read_file.assert_awaited_once_with(file="/tmp/test.txt")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file(started_backend: AioSandboxBackend) -> None:
    write_response = MagicMock()
    started_backend._client.file.write_file = AsyncMock(return_value=write_response)

    result = await started_backend.execute_tool("write_file", {"path": "/tmp/out.txt", "content": "hello"})

    assert result.success
    assert "Written" in result.output
    started_backend._client.file.write_file.assert_awaited_once_with(file="/tmp/out.txt", content="hello")


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_str_replace_success(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.replace_in_file = AsyncMock(return_value=_make_file_replace_response(1))

    result = await started_backend.execute_tool("str_replace", {"path": "/f.txt", "old_str": "bar", "new_str": "qux"})

    assert result.success
    started_backend._client.file.replace_in_file.assert_awaited_once_with(file="/f.txt", old_str="bar", new_str="qux")


@pytest.mark.asyncio
async def test_str_replace_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.replace_in_file = AsyncMock(return_value=_make_file_replace_response(0))

    result = await started_backend.execute_tool("str_replace", {"path": "/f.txt", "old_str": "missing", "new_str": "x"})

    assert not result.success
    assert result.exit_code == 1
    assert "not found" in (result.error or "")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_exception_returns_error(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(side_effect=ConnectionError("sandbox unreachable"))

    result = await started_backend.execute_tool("bash", {"command": "echo hi"})

    assert not result.success
    assert "sandbox unreachable" in (result.error or "")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_lifecycle() -> None:
    with (
        patch(
            "genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend.start", new_callable=AsyncMock
        ) as mock_start,
        patch("genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend.stop", new_callable=AsyncMock) as mock_stop,
    ):
        async with AioSandboxBackend():
            mock_start.assert_awaited_once()

        mock_stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# SandboxToolResult
# ---------------------------------------------------------------------------


def test_sandbox_tool_result_success_flag() -> None:
    assert SandboxToolResult(tool_name="bash", output="ok").success
    assert not SandboxToolResult(tool_name="bash", output="", exit_code=1, error="fail").success
    assert not SandboxToolResult(tool_name="bash", output="out", error="warning").success
