"""Integration tests for AioSandboxBackend — requires Docker.

Starts a real ``ghcr.io/agent-infra/sandbox`` container, runs all tools against it,
then stops the container.  Marked with ``integration`` so they are excluded from
the normal unit-test run.

Run with:
    uv run pytest tests/integration_tests/agents/test_sandbox_backend_integration.py -v -s
"""

import pytest
import pytest_asyncio
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)

from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend, AioSandboxBackendConfig

# Use a non-standard port to avoid conflicting with other services
TEST_PORT = 18091


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox():
    """Module-scoped fixture: one container for all tests."""
    backend = AioSandboxBackend(config=AioSandboxBackendConfig(host_port=TEST_PORT, startup_timeout=90.0))
    async with backend as b:
        yield b


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_bash_echo(sandbox: AioSandboxBackend) -> None:
    result = await sandbox.execute_tool("bash", {"command": "echo hello_world"})
    assert result.success, f"error: {result.error}"
    assert "hello_world" in result.output


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_bash_exit_code(sandbox: AioSandboxBackend) -> None:
    # Use a subshell so the session isn't terminated; the subshell exit code is propagated
    result = await sandbox.execute_tool("bash", {"command": "bash -c 'exit 42'"})
    assert result.exit_code == 42


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_bash_env_var(sandbox: AioSandboxBackend) -> None:
    result = await sandbox.execute_tool("bash", {"command": "echo $HOME"})
    assert result.success
    assert result.output.strip() != ""


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_write_and_read_file(sandbox: AioSandboxBackend) -> None:
    content = "integration test content\nline2\n"
    write_result = await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_test.txt", "content": content})
    assert write_result.success, f"write error: {write_result.error}"

    read_result = await sandbox.execute_tool("read_file", {"path": "/tmp/genai_tk_test.txt"})
    assert read_result.success, f"read error: {read_result.error}"
    assert read_result.output == content


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_str_replace(sandbox: AioSandboxBackend) -> None:
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_replace.txt", "content": "foo bar baz"})
    result = await sandbox.execute_tool(
        "str_replace",
        {"path": "/tmp/genai_tk_replace.txt", "old_str": "bar", "new_str": "QUX"},
    )
    assert result.success, f"replace error: {result.error}"

    read = await sandbox.execute_tool("read_file", {"path": "/tmp/genai_tk_replace.txt"})
    assert "QUX" in read.output
    assert "bar" not in read.output


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_str_replace_not_found(sandbox: AioSandboxBackend) -> None:
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_nf.txt", "content": "no match here"})
    result = await sandbox.execute_tool(
        "str_replace",
        {"path": "/tmp/genai_tk_nf.txt", "old_str": "MISSING_STRING", "new_str": "x"},
    )
    assert not result.success
    assert result.exit_code == 1


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_ls(sandbox: AioSandboxBackend) -> None:
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_ls_test.txt", "content": "ls test"})
    result = await sandbox.execute_tool("ls", {"path": "/tmp"})
    assert result.success, f"ls error: {result.error}"
    assert "genai_tk_ls_test.txt" in result.output


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_bash_multiline(sandbox: AioSandboxBackend) -> None:
    result = await sandbox.execute_tool("bash", {"command": "for i in 1 2 3; do echo item_$i; done"})
    assert result.success
    assert "item_1" in result.output
    assert "item_3" in result.output


# ---------------------------------------------------------------------------
# SandboxBackendProtocol — aexecute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aexecute_basic(sandbox: AioSandboxBackend) -> None:
    resp = await sandbox.aexecute("echo aexecute_ok")
    assert isinstance(resp, ExecuteResponse)
    assert resp.exit_code == 0
    assert "aexecute_ok" in resp.output


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aexecute_exit_code(sandbox: AioSandboxBackend) -> None:
    resp = await sandbox.aexecute("bash -c 'exit 7'")
    assert resp.exit_code == 7


# ---------------------------------------------------------------------------
# BackendProtocol — als_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_als_info_returns_file_info(sandbox: AioSandboxBackend) -> None:
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_ls_info.txt", "content": "x"})
    infos = await sandbox.als_info("/tmp")
    assert isinstance(infos, list)
    paths = [i["path"] for i in infos]
    assert any("genai_tk_ls_info.txt" in p for p in paths)


# ---------------------------------------------------------------------------
# BackendProtocol — aread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aread_line_numbered(sandbox: AioSandboxBackend) -> None:
    content = "alpha\nbeta\ngamma\n"
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_aread.txt", "content": content})

    result = await sandbox.aread("/tmp/genai_tk_aread.txt")

    assert result.error is None
    text = (result.file_data or {}).get("content", "")
    assert "1: alpha" in text
    assert "2: beta" in text
    assert "3: gamma" in text


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aread_pagination(sandbox: AioSandboxBackend) -> None:
    lines = "\n".join(f"L{i}" for i in range(1, 11))
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_tk_aread_pg.txt", "content": lines})

    result = await sandbox.aread("/tmp/genai_tk_aread_pg.txt", offset=2, limit=3)

    assert result.error is None
    text = (result.file_data or {}).get("content", "")
    assert "3: L3" in text
    assert "5: L5" in text
    assert "1: L1" not in text
    assert "6: L6" not in text


# ---------------------------------------------------------------------------
# BackendProtocol — awrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_awrite_creates_new_file(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_awrite_new.txt"
    # ensure file doesn't exist
    await sandbox.aexecute(f"rm -f {path}")

    result = await sandbox.awrite(path, "new file content")

    assert isinstance(result, WriteResult)
    assert result.error is None
    assert result.path == path

    read = await sandbox.aread(path)
    assert isinstance(read, ReadResult)
    assert read.error is None
    assert "new file content" in (read.file_data or {}).get("content", "")


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_awrite_fails_if_file_exists(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_awrite_exists.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "existing"})

    result = await sandbox.awrite(path, "should fail")

    assert result.error is not None
    assert "already exists" in result.error


# ---------------------------------------------------------------------------
# BackendProtocol — aedit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aedit_replace_first(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_aedit.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "aaa bbb aaa"})

    result = await sandbox.aedit(path, "aaa", "ZZZ", replace_all=False)

    assert isinstance(result, EditResult)
    assert result.error is None
    assert result.occurrences == 1

    read = await sandbox.aread(path)
    assert isinstance(read, ReadResult)
    assert read.error is None
    assert "ZZZ bbb aaa" in (read.file_data or {}).get("content", "")


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aedit_replace_all(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_aedit_all.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "x x x"})

    result = await sandbox.aedit(path, "x", "Y", replace_all=True)

    assert result.occurrences == 3
    read = await sandbox.aread(path)
    assert isinstance(read, ReadResult)
    assert read.error is None
    assert "Y Y Y" in (read.file_data or {}).get("content", "")


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aedit_string_not_found(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_aedit_nf.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "hello world"})

    result = await sandbox.aedit(path, "MISSING", "x")

    assert result.error is not None


# ---------------------------------------------------------------------------
# BackendProtocol — agrep_raw
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_agrep_raw_finds_pattern(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_grep.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "hello world\nfoo bar\nhello again"})

    result = await sandbox.agrep_raw("hello", path="/tmp", glob="genai_tk_grep.txt")

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(m, dict) and "path" in m and "line" in m and "text" in m for m in result)
    assert result[0]["line"] == 1
    assert result[1]["line"] == 3


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_agrep_raw_no_matches(sandbox: AioSandboxBackend) -> None:
    path = "/tmp/genai_tk_grep_nm.txt"
    await sandbox.execute_tool("write_file", {"path": path, "content": "no match here"})

    result = await sandbox.agrep_raw("XYZNOTFOUND", path="/tmp", glob="genai_tk_grep_nm.txt")

    assert result == []


# ---------------------------------------------------------------------------
# BackendProtocol — aglob_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_aglob_info_matches_files(sandbox: AioSandboxBackend) -> None:
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_glob_a.py", "content": "# py"})
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_glob_b.py", "content": "# py"})
    await sandbox.execute_tool("write_file", {"path": "/tmp/genai_glob_c.txt", "content": "txt"})

    infos = await sandbox.aglob_info("genai_glob_*.py", path="/tmp")

    paths = [i["path"] for i in infos]
    assert any("genai_glob_a.py" in p for p in paths)
    assert any("genai_glob_b.py" in p for p in paths)
    assert not any("genai_glob_c.txt" in p for p in paths)


# ---------------------------------------------------------------------------
# BackendProtocol — aupload_files / adownload_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_upload_and_download_files(sandbox: AioSandboxBackend) -> None:
    files = [
        ("/tmp/genai_tk_upload_a.txt", b"content_a"),
        ("/tmp/genai_tk_upload_b.txt", b"content_b"),
    ]

    up = await sandbox.aupload_files(files)
    assert all(isinstance(r, FileUploadResponse) for r in up)
    assert all(r.error is None for r in up)

    down = await sandbox.adownload_files(["/tmp/genai_tk_upload_a.txt", "/tmp/genai_tk_upload_b.txt"])
    assert all(isinstance(r, FileDownloadResponse) for r in down)
    assert all(r.error is None for r in down)
    assert down[0].content == b"content_a"
    assert down[1].content == b"content_b"


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.integration
async def test_download_missing_file(sandbox: AioSandboxBackend) -> None:
    down = await sandbox.adownload_files(["/tmp/genai_tk_does_not_exist.txt"])
    assert down[0].error == "file_not_found"
