"""Integration tests for AioSandboxBackend — requires Docker.

Starts a real ``ghcr.io/agent-infra/sandbox`` container, runs all tools against it,
then stops the container.  Marked with ``integration`` so they are excluded from
the normal unit-test run.

Run with:
    uv run pytest tests/integration_tests/agents/test_sandbox_backend_integration.py -v -s
"""

import pytest
import pytest_asyncio

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
