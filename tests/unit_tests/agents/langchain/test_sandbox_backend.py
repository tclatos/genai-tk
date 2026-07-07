"""Unit tests for AioSandboxBackend — all mocked, no Docker required.

The backend delegates shell/file work to the native ``opensandbox`` SDK
(``sandbox.commands.run`` / ``sandbox.files.*``); these tests mock that
surface so no container or server is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.config_mgmt.features import is_available

if not is_available("harnessing"):
    pytest.skip(
        "Optional feature 'harnessing' not installed — run: uv sync --extra harnessing", allow_module_level=True
    )

from deepagents.backends.protocol import (  # noqa: E402
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from genai_tk.agents.langchain.sandbox_backend import (  # noqa: E402
    AioSandboxBackend,
    AioSandboxBackendConfig,
    SandboxToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_execution(text: str = "", exit_code: int = 0) -> MagicMock:
    """Build a mock ``opensandbox.Execution`` with ``.text`` and ``.exit_code``."""
    execution = MagicMock()
    execution.text = text
    execution.exit_code = exit_code
    logs = MagicMock()
    logs.stderr = []
    execution.logs = logs
    return execution


def _make_entry(path: str, size: int | None = None) -> MagicMock:
    """Build a mock ``opensandbox.EntryInfo`` with ``.path`` and ``.size``."""
    entry = MagicMock()
    entry.path = path
    entry.size = size
    entry.entry_type = None
    return entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> AioSandboxBackend:
    return AioSandboxBackend(config=AioSandboxBackendConfig(startup_timeout=1.0))


@pytest.fixture
def started_backend(backend: AioSandboxBackend) -> AioSandboxBackend:
    """Backend with a mock opensandbox Sandbox already connected."""
    sandbox = MagicMock()
    sandbox.commands = MagicMock()
    sandbox.commands.run = AsyncMock()
    sandbox.files = MagicMock()
    sandbox.files.read_file = AsyncMock()
    sandbox.files.read_bytes = AsyncMock()
    sandbox.files.write_file = AsyncMock()
    sandbox.files.search = AsyncMock()
    backend._sandbox = sandbox
    return backend


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_list_tools(backend: AioSandboxBackend) -> None:
    assert set(backend.list_tools()) == {"bash", "ls", "read_file", "write_file", "str_replace"}


def test_build_browser_env_clears_image_default_user_agent(backend: AioSandboxBackend) -> None:
    env = backend._build_browser_env()

    assert env["BROWSER_USER_AGENT"] == ""
    assert env["TZ"] == "Europe/Paris"
    assert env["LANG"] == "fr_FR.UTF-8"
    assert env["LC_ALL"] == "fr_FR.UTF-8"
    assert env["LANGUAGE"] == "fr_FR:fr"
    assert env["BROWSER_EXTRA_ARGS"] == (
        "--lang=fr-FR --time-zone-for-testing=Europe/Paris --disable-blink-features=AutomationControlled"
    )


def test_build_browser_env_preserves_explicit_browser_overrides() -> None:
    backend = AioSandboxBackend(
        config=AioSandboxBackendConfig(
            startup_timeout=1.0,
            env_vars={
                "BROWSER_USER_AGENT": "custom-agent",
                "TZ": "UTC",
                "LANG": "de_DE.UTF-8",
                "LC_ALL": "de_DE.UTF-8",
                "LANGUAGE": "de_DE:de",
                "BROWSER_EXTRA_ARGS": (
                    "--foo=bar --lang=de-DE --time-zone-for-testing=Europe/Berlin "
                    "--disable-blink-features=AutomationControlled"
                ),
            },
        )
    )

    env = backend._build_browser_env()

    assert env["BROWSER_USER_AGENT"] == "custom-agent"
    assert env["TZ"] == "UTC"
    assert env["LANG"] == "de_DE.UTF-8"
    assert env["LC_ALL"] == "de_DE.UTF-8"
    assert env["LANGUAGE"] == "de_DE:de"
    assert env["BROWSER_EXTRA_ARGS"] == (
        "--foo=bar --lang=de-DE --time-zone-for-testing=Europe/Berlin --disable-blink-features=AutomationControlled"
    )


async def test_execute_tool_not_started(backend: AioSandboxBackend) -> None:
    with pytest.raises(RuntimeError, match="not started"):
        await backend.execute_tool("bash", {"command": "echo hi"})


async def test_execute_unknown_tool(started_backend: AioSandboxBackend) -> None:
    with pytest.raises(ValueError, match="Unsupported tool"):
        await started_backend.execute_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


async def test_bash_success(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("hello\n", 0))

    result = await started_backend.execute_tool("bash", {"command": "echo hello"})

    assert result.success
    assert result.output == "hello\n"
    assert result.tool_name == "bash"
    started_backend._sandbox.commands.run.assert_awaited_once_with("echo hello")


async def test_bash_failure(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 1))

    result = await started_backend.execute_tool("bash", {"command": "bogus"})

    assert not result.success
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


async def test_ls_uses_path(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("file.txt\n", 0))

    result = await started_backend.execute_tool("ls", {"path": "/tmp"})

    assert result.success
    assert "file.txt" in result.output
    started_backend._sandbox.commands.run.assert_awaited_once_with("ls -1pA /tmp 2>/dev/null")


async def test_ls_defaults_to_work_dir(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 0))

    result = await started_backend.execute_tool("ls", {})

    assert result.success
    started_backend._sandbox.commands.run.assert_awaited_once_with("ls -1pA /home/user 2>/dev/null")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def test_read_file(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="file content")

    result = await started_backend.execute_tool("read_file", {"path": "/tmp/test.txt"})

    assert result.success
    assert result.output == "file content"
    started_backend._sandbox.files.read_file.assert_awaited_once_with("/tmp/test.txt")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_file(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.execute_tool("write_file", {"path": "/tmp/out.txt", "content": "hello"})

    assert result.success
    assert "Written" in result.output
    started_backend._sandbox.files.write_file.assert_awaited_once_with("/tmp/out.txt", "hello")


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------


async def test_str_replace_success(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="foo bar baz")
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.execute_tool("str_replace", {"path": "/f.txt", "old_str": "bar", "new_str": "qux"})

    assert result.success
    started_backend._sandbox.files.write_file.assert_awaited_once_with("/f.txt", "foo qux baz")


async def test_str_replace_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="no match here")
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.execute_tool("str_replace", {"path": "/f.txt", "old_str": "missing", "new_str": "x"})

    assert not result.success
    assert result.exit_code == 1
    assert "not found" in (result.error or "")
    started_backend._sandbox.files.write_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_execute_tool_exception_returns_error(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(side_effect=ConnectionError("sandbox unreachable"))

    result = await started_backend.execute_tool("bash", {"command": "echo hi"})

    assert not result.success
    assert "sandbox unreachable" in (result.error or "")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


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


async def test_aexecute_success(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("out\n", 0))

    resp = await started_backend.aexecute("echo out")

    assert isinstance(resp, ExecuteResponse)
    assert resp.output == "out\n"
    assert resp.exit_code == 0
    started_backend._sandbox.commands.run.assert_awaited_once_with("echo out")


async def test_aexecute_nonzero_exit(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 1))

    resp = await started_backend.aexecute("false")

    assert resp.exit_code == 1


# ---------------------------------------------------------------------------
# BackendProtocol.als
# ---------------------------------------------------------------------------


async def test_als_returns_ls_result(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("a.py\nb.txt\n", 0))

    result = await started_backend.als("/home/user")

    assert isinstance(result, LsResult)
    assert result.error is None
    assert result.entries is not None
    assert len(result.entries) == 2
    assert result.entries[0]["path"] == "/home/user/a.py"
    assert result.entries[1]["path"] == "/home/user/b.txt"


async def test_als_marks_directories(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("a.py\nsub/\n", 0))

    result = await started_backend.als("/home/user")

    assert result.entries is not None
    by_name = {e["path"].split("/")[-1]: e for e in result.entries}
    assert by_name["sub"]["is_dir"] is True
    assert "is_dir" not in by_name["a.py"]


async def test_als_empty_directory(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 0))

    result = await started_backend.als("/empty")

    assert isinstance(result, LsResult)
    assert result.entries == []


# ---------------------------------------------------------------------------
# BackendProtocol.aread
# ---------------------------------------------------------------------------


async def test_aread_returns_numbered_lines(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="line1\nline2\nline3\n")

    result = await started_backend.aread("/tmp/f.txt")

    assert isinstance(result, ReadResult)
    assert result.error is None
    assert result.file_data is not None
    assert "1: line1\n" in result.file_data["content"]
    assert "2: line2\n" in result.file_data["content"]
    assert "3: line3\n" in result.file_data["content"]


async def test_aread_pagination_offset_limit(started_backend: AioSandboxBackend) -> None:
    content = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
    started_backend._sandbox.files.read_file = AsyncMock(return_value=content)

    result = await started_backend.aread("/tmp/f.txt", offset=2, limit=3)

    assert isinstance(result, ReadResult)
    assert result.file_data is not None
    text = result.file_data["content"]
    assert "3: line3" in text
    assert "4: line4" in text
    assert "5: line5" in text
    assert "1: line1" not in text
    assert "6: line6" not in text


async def test_aread_error_returns_error_string(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(side_effect=RuntimeError("not found"))

    result = await started_backend.aread("/missing.txt")

    assert isinstance(result, ReadResult)
    assert result.error is not None
    assert result.error.startswith("Error:")
    assert result.file_data is None


# ---------------------------------------------------------------------------
# BackendProtocol.awrite
# ---------------------------------------------------------------------------


async def test_awrite_new_file_success(started_backend: AioSandboxBackend) -> None:
    # bash existence check returns ABSENT
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("ABSENT\n", 0))
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.awrite("/tmp/new.txt", "content")

    assert isinstance(result, WriteResult)
    assert result.error is None
    assert result.path == "/tmp/new.txt"
    started_backend._sandbox.files.write_file.assert_awaited_once_with("/tmp/new.txt", "content")


async def test_awrite_existing_file_returns_error(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("EXISTS\n", 0))

    result = await started_backend.awrite("/tmp/exists.txt", "content")

    assert result.error is not None
    assert "already exists" in result.error
    started_backend._sandbox.files.write_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# BackendProtocol.aedit
# ---------------------------------------------------------------------------


async def test_aedit_replaces_first_occurrence(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="foo foo foo")
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.aedit("/f.txt", "foo", "bar", replace_all=False)

    assert isinstance(result, EditResult)
    assert result.error is None
    assert result.occurrences == 1
    started_backend._sandbox.files.write_file.assert_awaited_once_with("/f.txt", "bar foo foo")


async def test_aedit_replaces_all_occurrences(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="foo foo foo")
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.aedit("/f.txt", "foo", "bar", replace_all=True)

    assert result.occurrences == 3
    started_backend._sandbox.files.write_file.assert_awaited_once_with("/f.txt", "bar bar bar")


async def test_aedit_string_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(return_value="no match here")
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

    result = await started_backend.aedit("/f.txt", "missing", "x")

    assert result.error is not None
    assert "not found" in result.error
    started_backend._sandbox.files.write_file.assert_not_awaited()


async def test_aedit_read_error(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_file = AsyncMock(side_effect=RuntimeError("no such file"))

    result = await started_backend.aedit("/missing.txt", "a", "b")

    assert result.error is not None
    assert "Cannot read" in result.error


# ---------------------------------------------------------------------------
# BackendProtocol.agrep
# ---------------------------------------------------------------------------


async def test_agrep_returns_matches(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(
        return_value=_make_execution("/src/a.py:10:    foo = bar\n/src/b.py:42:    foo = baz", 0)
    )

    result = await started_backend.agrep("foo", path="/src")

    assert isinstance(result, GrepResult)
    assert result.error is None
    assert result.matches is not None
    assert len(result.matches) == 2
    assert result.matches[0] == GrepMatch(path="/src/a.py", line=10, text="    foo = bar")
    assert result.matches[1] == GrepMatch(path="/src/b.py", line=42, text="    foo = baz")


async def test_agrep_no_matches_returns_empty_list(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 1))

    result = await started_backend.agrep("notfound", path="/src")

    assert isinstance(result, GrepResult)
    assert result.matches == []
    assert result.error is None


async def test_agrep_error_returns_grep_result_with_error(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("grep: bad", 2))

    result = await started_backend.agrep("x", path="/bad")

    assert isinstance(result, GrepResult)
    assert result.error is not None
    assert "grep error" in result.error


async def test_agrep_with_glob_passes_include(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("", 1))

    await started_backend.agrep("foo", path="/src", glob="*.py")

    command = started_backend._sandbox.commands.run.call_args.args[0]
    assert "--include=" in command


# ---------------------------------------------------------------------------
# BackendProtocol.aglob
# ---------------------------------------------------------------------------


async def test_aglob_returns_glob_result(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.search = AsyncMock(
        return_value=[_make_entry("/src/a.py", 100), _make_entry("/src/b.py", 200)]
    )

    result = await started_backend.aglob("*.py", path="/src")

    assert isinstance(result, GlobResult)
    assert result.error is None
    assert result.matches is not None
    assert all(isinstance(i, dict) and "path" in i for i in result.matches)
    assert result.matches[0]["path"] == "/src/a.py"
    assert result.matches[0]["size"] == 100
    assert result.matches[1]["path"] == "/src/b.py"


async def test_aglob_empty_result(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.search = AsyncMock(return_value=[])

    result = await started_backend.aglob("*.nonexistent", path="/src")

    assert isinstance(result, GlobResult)
    assert result.matches == []


# ---------------------------------------------------------------------------
# BackendProtocol.aupload_files
# ---------------------------------------------------------------------------


async def test_aupload_files_success(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.write_file = AsyncMock(return_value=None)

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


async def test_aupload_files_write_error_gives_permission_denied(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.write_file = AsyncMock(side_effect=RuntimeError("forbidden"))

    responses = await started_backend.aupload_files([("/bad.txt", b"x")])

    assert responses[0].error == "permission_denied"


# ---------------------------------------------------------------------------
# BackendProtocol.adownload_files
# ---------------------------------------------------------------------------


async def test_adownload_files_success(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_bytes = AsyncMock(return_value=b"file content")

    responses = await started_backend.adownload_files(["/a.txt", "/b.txt"])

    assert len(responses) == 2
    assert all(isinstance(r, FileDownloadResponse) for r in responses)
    assert all(r.error is None for r in responses)
    assert responses[0].content == b"file content"


async def test_adownload_files_missing_gives_file_not_found(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.read_bytes = AsyncMock(side_effect=RuntimeError("not found"))

    responses = await started_backend.adownload_files(["/missing.txt"])

    assert responses[0].error == "file_not_found"
    assert responses[0].content is None


# ---------------------------------------------------------------------------
# Deprecated SandboxBackendProtocol methods (als_info / agrep_raw / aglob_info)
# ---------------------------------------------------------------------------
# These are deprecated in deepagents but still part of SandboxBackendProtocol
# and may be called by either harness. They delegate to als/agrep/aglob by
# default; verify the delegation so both harness paths stay supported.


@pytest.mark.filterwarnings("ignore::langchain_core._api.deprecation.LangChainDeprecationWarning")
async def test_als_info_delegates_to_als(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("a.py\nsub/\n", 0))

    infos = await started_backend.als_info("/home/user")

    assert isinstance(infos, list)
    assert any(i["path"] == "/home/user/a.py" for i in infos)


@pytest.mark.filterwarnings("ignore::langchain_core._api.deprecation.LangChainDeprecationWarning")
async def test_agrep_raw_delegates_to_agrep(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("/src/a.py:10:    foo = bar\n", 0))

    out = await started_backend.agrep_raw("foo", path="/src")

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == GrepMatch(path="/src/a.py", line=10, text="    foo = bar")


@pytest.mark.filterwarnings("ignore::langchain_core._api.deprecation.LangChainDeprecationWarning")
async def test_agrep_raw_returns_error_string_on_agrep_error(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.commands.run = AsyncMock(return_value=_make_execution("grep: bad", 2))

    out = await started_backend.agrep_raw("x", path="/bad")

    assert isinstance(out, str)
    assert "grep error" in out


@pytest.mark.filterwarnings("ignore::langchain_core._api.deprecation.LangChainDeprecationWarning")
async def test_aglob_info_delegates_to_aglob(started_backend: AioSandboxBackend) -> None:
    started_backend._sandbox.files.search = AsyncMock(return_value=[_make_entry("/src/a.py"), _make_entry("/src/b.py")])

    infos = await started_backend.aglob_info("*.py", path="/src")

    assert isinstance(infos, list)
    assert all(isinstance(i, dict) and "path" in i for i in infos)
    assert infos[0]["path"] == "/src/a.py"
    assert infos[1]["path"] == "/src/b.py"
