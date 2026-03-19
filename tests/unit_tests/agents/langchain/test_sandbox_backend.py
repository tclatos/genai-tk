"""Unit tests for AioSandboxBackend — all mocked, no Docker required."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)

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


# ---------------------------------------------------------------------------
# SandboxBackendProtocol.id
# ---------------------------------------------------------------------------


def test_id_returns_instance_id_when_not_started(backend: AioSandboxBackend) -> None:
    assert len(backend.id) == 12  # _instance_id hex


def test_id_no_sandbox_falls_back_to_instance_id(backend: AioSandboxBackend) -> None:
    assert backend._sandbox is None
    assert len(backend.id) == 12


# ---------------------------------------------------------------------------
# SandboxBackendProtocol.aexecute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aexecute_success(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("out\n", 0))

    resp = await started_backend.aexecute("echo out")

    assert isinstance(resp, ExecuteResponse)
    assert resp.output == "out\n"
    assert resp.exit_code == 0
    started_backend._client.shell.exec_command.assert_awaited_once_with(command="echo out")


@pytest.mark.asyncio
async def test_aexecute_nonzero_exit(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("", 1))

    resp = await started_backend.aexecute("false")

    assert resp.exit_code == 1


# ---------------------------------------------------------------------------
# BackendProtocol.als_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_als_info_returns_file_info_list(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.list_path = AsyncMock(
        return_value=_make_file_list_response([("/home/user/a.py", 100), ("/home/user/b.txt", None)])
    )

    infos = await started_backend.als_info("/home/user")

    assert len(infos) == 2
    assert infos[0]["path"] == "/home/user/a.py"
    assert infos[0]["size"] == 100
    assert infos[1]["path"] == "/home/user/b.txt"
    assert "size" not in infos[1]


@pytest.mark.asyncio
async def test_als_info_empty_directory(started_backend: AioSandboxBackend) -> None:
    data = MagicMock()
    data.files = []
    resp = MagicMock()
    resp.data = data
    started_backend._client.file.list_path = AsyncMock(return_value=resp)

    infos = await started_backend.als_info("/empty")

    assert infos == []


# ---------------------------------------------------------------------------
# BackendProtocol.aread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aread_returns_numbered_lines(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("line1\nline2\nline3\n"))

    text = await started_backend.aread("/tmp/f.txt")

    assert "1: line1\n" in text
    assert "2: line2\n" in text
    assert "3: line3\n" in text


@pytest.mark.asyncio
async def test_aread_pagination_offset_limit(started_backend: AioSandboxBackend) -> None:
    content = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response(content))

    text = await started_backend.aread("/tmp/f.txt", offset=2, limit=3)

    assert "3: line3" in text
    assert "4: line4" in text
    assert "5: line5" in text
    assert "1: line1" not in text
    assert "6: line6" not in text


@pytest.mark.asyncio
async def test_aread_error_returns_error_string(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(side_effect=RuntimeError("not found"))

    text = await started_backend.aread("/missing.txt")

    assert text.startswith("Error:")


# ---------------------------------------------------------------------------
# BackendProtocol.awrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awrite_new_file_success(started_backend: AioSandboxBackend) -> None:
    # bash check returns ABSENT
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("ABSENT\n", 0))
    started_backend._client.file.write_file = AsyncMock(return_value=MagicMock())

    result = await started_backend.awrite("/tmp/new.txt", "content")

    assert isinstance(result, WriteResult)
    assert result.error is None
    assert result.path == "/tmp/new.txt"


@pytest.mark.asyncio
async def test_awrite_existing_file_returns_error(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("EXISTS\n", 0))

    result = await started_backend.awrite("/tmp/exists.txt", "content")

    assert result.error is not None
    assert "already exists" in result.error
    started_backend._client.file.write_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# BackendProtocol.aedit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aedit_replaces_first_occurrence(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("foo foo foo"))
    started_backend._client.file.write_file = AsyncMock(return_value=MagicMock())

    result = await started_backend.aedit("/f.txt", "foo", "bar", replace_all=False)

    assert isinstance(result, EditResult)
    assert result.error is None
    assert result.occurrences == 1
    started_backend._client.file.write_file.assert_awaited_once_with(file="/f.txt", content="bar foo foo")


@pytest.mark.asyncio
async def test_aedit_replaces_all_occurrences(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("foo foo foo"))
    started_backend._client.file.write_file = AsyncMock(return_value=MagicMock())

    result = await started_backend.aedit("/f.txt", "foo", "bar", replace_all=True)

    assert result.occurrences == 3
    started_backend._client.file.write_file.assert_awaited_once_with(file="/f.txt", content="bar bar bar")


@pytest.mark.asyncio
async def test_aedit_string_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("no match here"))

    result = await started_backend.aedit("/f.txt", "missing", "x")

    assert result.error is not None
    assert "not found" in result.error
    started_backend._client.file.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_aedit_read_error(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(side_effect=RuntimeError("no such file"))

    result = await started_backend.aedit("/missing.txt", "a", "b")

    assert result.error is not None
    assert "Cannot read" in result.error


# ---------------------------------------------------------------------------
# BackendProtocol.agrep_raw
# ---------------------------------------------------------------------------


def _make_grep_shell_response(lines: list[str], exit_code: int = 0) -> MagicMock:
    return _make_shell_response("\n".join(lines), exit_code)


@pytest.mark.asyncio
async def test_agrep_raw_returns_matches(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(
        return_value=_make_grep_shell_response(["/src/a.py:10:    foo = bar", "/src/b.py:42:    foo = baz"])
    )

    result = await started_backend.agrep_raw("foo", path="/src")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == GrepMatch(path="/src/a.py", line=10, text="    foo = bar")
    assert result[1] == GrepMatch(path="/src/b.py", line=42, text="    foo = baz")


@pytest.mark.asyncio
async def test_agrep_raw_no_matches_returns_empty_list(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("", 1))

    result = await started_backend.agrep_raw("notfound", path="/src")

    assert result == []


@pytest.mark.asyncio
async def test_agrep_raw_error_returns_string(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("grep: bad", 2))

    result = await started_backend.agrep_raw("x", path="/bad")

    assert isinstance(result, str)
    assert "grep error" in result


@pytest.mark.asyncio
async def test_agrep_raw_with_glob_passes_include(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("", 1))

    await started_backend.agrep_raw("foo", path="/src", glob="*.py")

    call_args = started_backend._client.shell.exec_command.call_args
    assert "--include=" in call_args.kwargs["command"] or "--include=" in str(call_args)


# ---------------------------------------------------------------------------
# BackendProtocol.aglob_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aglob_info_returns_file_infos(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(
        return_value=_make_shell_response("/src/a.py\n/src/b.py\n", 0)
    )

    infos = await started_backend.aglob_info("*.py", path="/src")

    assert isinstance(infos, list)
    assert all(isinstance(i, dict) and "path" in i for i in infos)
    assert infos[0]["path"] == "/src/a.py"
    assert infos[1]["path"] == "/src/b.py"


@pytest.mark.asyncio
async def test_aglob_info_empty_result(started_backend: AioSandboxBackend) -> None:
    started_backend._client.shell.exec_command = AsyncMock(return_value=_make_shell_response("", 0))

    infos = await started_backend.aglob_info("*.nonexistent", path="/src")

    assert infos == []


# ---------------------------------------------------------------------------
# BackendProtocol.aupload_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aupload_files_success(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.write_file = AsyncMock(return_value=MagicMock())

    responses = await started_backend.aupload_files(
        [
            ("/a.txt", b"hello"),
            ("/b.txt", b"world"),
        ]
    )

    assert len(responses) == 2
    assert all(isinstance(r, FileUploadResponse) for r in responses)
    assert all(r.error is None for r in responses)
    assert responses[0].path == "/a.txt"
    assert responses[1].path == "/b.txt"


@pytest.mark.asyncio
async def test_aupload_files_write_error_gives_permission_denied(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.write_file = AsyncMock(side_effect=RuntimeError("forbidden"))

    responses = await started_backend.aupload_files([("/bad.txt", b"x")])

    assert responses[0].error == "permission_denied"


# ---------------------------------------------------------------------------
# BackendProtocol.adownload_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adownload_files_success(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(return_value=_make_file_read_response("file content"))

    responses = await started_backend.adownload_files(["/a.txt", "/b.txt"])

    assert len(responses) == 2
    assert all(isinstance(r, FileDownloadResponse) for r in responses)
    assert all(r.error is None for r in responses)
    assert responses[0].content == b"file content"


@pytest.mark.asyncio
async def test_adownload_files_missing_gives_file_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._client.file.read_file = AsyncMock(side_effect=RuntimeError("not found"))

    responses = await started_backend.adownload_files(["/missing.txt"])

    assert responses[0].error == "file_not_found"
    assert responses[0].content is None
