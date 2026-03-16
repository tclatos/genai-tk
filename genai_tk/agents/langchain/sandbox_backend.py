"""LangChain deepagents BackendProtocol backed by agent-infra/sandbox.

Uses ``agent_sandbox.AsyncSandbox`` as an HTTP client against a running Docker
container (``ghcr.io/agent-infra/sandbox``).  This module manages the full
container lifecycle: starts the container on ``start()``, polls until the HTTP
API is healthy, then stops + removes the container on ``stop()``.

The tool implementations mirror Deer-flow's own ``AioSandbox`` class:
``shell.exec_command`` for bash/ls, ``file.*`` endpoints for file I/O.

``AioSandboxBackend`` implements ``SandboxBackendProtocol`` (which extends
``BackendProtocol``) from deepagents, providing:

- ``execute_tool`` — low-level named-tool dispatch (bash, ls, read_file, write_file, str_replace)
- ``aexecute`` — shell command → ``ExecuteResponse``
- ``als_info`` / ``aread`` / ``awrite`` / ``aedit`` — file operations → typed results
- ``agrep_raw`` / ``aglob_info`` — search operations
- ``aupload_files`` / ``adownload_files`` — bulk file I/O

Example:
```python
from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

async with AioSandboxBackend() as backend:
    result = await backend.execute_tool("bash", {"command": "echo hello"})
    print(result.output)

    # BackendProtocol interface
    resp = await backend.aexecute("ls /tmp")
    print(resp.output)

    files = await backend.als_info("/home/user")
    print([f["path"] for f in files])
```
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import uuid
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from agent_sandbox import AsyncSandbox


SANDBOX_IMAGE = "ghcr.io/agent-infra/sandbox:latest"
SANDBOX_INTERNAL_PORT = 8091  # SANDBOX_SRV_PORT — the actual REST API port

_SUPPORTED_TOOLS = frozenset({"bash", "ls", "read_file", "write_file", "str_replace"})


class SandboxToolResult(BaseModel):
    """Result of a sandbox tool execution."""

    tool_name: str
    output: str
    exit_code: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.error is None


class AioSandboxBackendConfig(BaseModel):
    """Configuration for ``AioSandboxBackend``."""

    image: str = SANDBOX_IMAGE
    host: str = "127.0.0.1"
    host_port: int = 18091
    startup_timeout: float = 60.0
    work_dir: str = "/home/user"
    env_vars: dict[str, str] = Field(default_factory=dict)


class AioSandboxBackend(SandboxBackendProtocol, BaseModel):
    """deepagents ``SandboxBackendProtocol`` backed by ``agent_sandbox.AsyncSandbox``.

    Manages the Docker container lifecycle: starts on ``__aenter__``, exposes
    the full ``BackendProtocol`` interface plus the low-level ``execute_tool``
    dispatch, and stops the container on ``__aexit__``.

    Async-native: the ``a*`` protocol methods are the primary implementations;
    sync counterparts (``ls_info``, ``read``, etc.) raise ``NotImplementedError``
    since the backend depends on a running event loop.
    """

    config: AioSandboxBackendConfig = Field(default_factory=AioSandboxBackendConfig)

    _container_id: str | None = None
    _client: AsyncSandbox | None = None
    _instance_id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex[:12])

    model_config = {"arbitrary_types_allowed": True}

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.host_port}"

    @property
    def id(self) -> str:
        """Unique identifier: container short ID when running, otherwise a random hex."""
        if self._container_id:
            return self._container_id[:12]
        return self._instance_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the sandbox Docker container and wait until healthy."""
        try:
            from agent_sandbox import AsyncSandbox  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("agent-sandbox is required: uv add agent-sandbox") from exc

        docker_cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{self.config.host_port}:{SANDBOX_INTERNAL_PORT}",
        ]
        for k, v in self.config.env_vars.items():
            docker_cmd += ["-e", f"{k}={v}"]
        docker_cmd.append(self.config.image)

        logger.debug(f"Starting sandbox container: {self.config.image}")
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {stderr.decode().strip()}")

        self._container_id = stdout.decode().strip()
        logger.debug(f"Container started: {self._container_id[:12]}")

        # Pass trust_env=False so http_proxy env vars don't route localhost through a corporate proxy
        import httpx  # noqa: PLC0415

        httpx_client = httpx.AsyncClient(trust_env=False, timeout=120)
        self._client = AsyncSandbox(base_url=self.base_url, httpx_client=httpx_client)
        await self._wait_healthy()
        logger.info(f"AioSandbox ready at {self.base_url}")

    async def stop(self) -> None:
        """Stop and remove the sandbox Docker container."""
        self._client = None
        if self._container_id:
            cid = self._container_id
            self._container_id = None
            logger.debug(f"Stopping container {cid[:12]}")
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "stop",
                cid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.communicate()
            logger.info("AioSandbox stopped")

    async def __aenter__(self) -> AioSandboxBackend:
        await self.start()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # BackendProtocol interface
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return the tool names supported by this backend."""
        return sorted(_SUPPORTED_TOOLS)

    async def execute_tool(self, tool_name: str, tool_input: dict) -> SandboxToolResult:
        """Execute a named tool inside the sandbox container.

        Args:
            tool_name: One of ``bash``, ``ls``, ``read_file``, ``write_file``, ``str_replace``.
            tool_input: Tool-specific parameters.

        Returns:
            ``SandboxToolResult`` with ``output``, ``exit_code``, and optional ``error``.
        """
        if self._client is None:
            raise RuntimeError("Backend not started — use 'async with AioSandboxBackend()' or call start() first")
        if tool_name not in _SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool '{tool_name}'. Available: {sorted(_SUPPORTED_TOOLS)}")

        try:
            match tool_name:
                case "bash":
                    return await self._run_bash(tool_input)
                case "ls":
                    return await self._run_ls(tool_input)
                case "read_file":
                    return await self._run_read_file(tool_input)
                case "write_file":
                    return await self._run_write_file(tool_input)
                case "str_replace":
                    return await self._run_str_replace(tool_input)
                case _:  # pragma: no cover
                    raise ValueError(f"Unhandled tool: {tool_name}")
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' raised: {exc}")
            return SandboxToolResult(tool_name=tool_name, output="", exit_code=1, error=str(exc))

    # ------------------------------------------------------------------
    # Tool implementations — mirror Deer-flow's AioSandbox
    # ------------------------------------------------------------------

    async def _run_bash(self, tool_input: dict) -> SandboxToolResult:
        assert self._client is not None
        command: str = tool_input["command"]
        resp = await self._client.shell.exec_command(command=command)
        result = resp.data
        output = (result.output or "") if result else ""
        exit_code = result.exit_code if result and result.exit_code is not None else 0
        return SandboxToolResult(tool_name="bash", output=output, exit_code=exit_code)

    async def _run_ls(self, tool_input: dict) -> SandboxToolResult:
        assert self._client is not None
        path: str = tool_input.get("path", self.config.work_dir)
        resp = await self._client.file.list_path(path=path, include_size=True, show_hidden=True)
        result = resp.data
        if result and result.files:
            lines = [f.path + (f"  ({f.size}B)" if f.size is not None else "") for f in result.files]
            output = "\n".join(lines)
        else:
            output = ""
        return SandboxToolResult(tool_name="ls", output=output)

    async def _run_read_file(self, tool_input: dict) -> SandboxToolResult:
        assert self._client is not None
        file_path: str = tool_input["path"]
        resp = await self._client.file.read_file(file=file_path)
        result = resp.data
        content = result.content if result else ""
        return SandboxToolResult(tool_name="read_file", output=content)

    async def _run_write_file(self, tool_input: dict) -> SandboxToolResult:
        assert self._client is not None
        file_path: str = tool_input["path"]
        content: str = tool_input["content"]
        await self._client.file.write_file(file=file_path, content=content)
        return SandboxToolResult(tool_name="write_file", output=f"Written: {file_path}")

    async def _run_str_replace(self, tool_input: dict) -> SandboxToolResult:
        assert self._client is not None
        file_path: str = tool_input["path"]
        old_str: str = tool_input["old_str"]
        new_str: str = tool_input["new_str"]
        resp = await self._client.file.replace_in_file(file=file_path, old_str=old_str, new_str=new_str)
        result = resp.data
        if result and (result.replaced_count or 0) > 0:
            return SandboxToolResult(
                tool_name="str_replace",
                output=f"Replaced {result.replaced_count}x in: {file_path}",
            )
        return SandboxToolResult(
            tool_name="str_replace",
            output="",
            exit_code=1,
            error=f"String not found in {file_path}",
        )

    # ------------------------------------------------------------------
    # SandboxBackendProtocol — aexecute
    # ------------------------------------------------------------------

    async def aexecute(  # noqa: ASYNC109
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command and return a structured ``ExecuteResponse``.

        Args:
            command: Shell command to run inside the sandbox.
            timeout: Not enforced by this backend (ignored).

        Returns:
            ``ExecuteResponse`` with combined output and exit code.
        """
        result = await self._run_bash({"command": command})
        return ExecuteResponse(output=result.output, exit_code=result.exit_code)

    # ------------------------------------------------------------------
    # BackendProtocol — file operations
    # ------------------------------------------------------------------

    async def als_info(self, path: str) -> list[FileInfo]:
        """List directory contents with metadata.

        Args:
            path: Absolute path to the directory.

        Returns:
            List of ``FileInfo`` dicts with ``path`` and optional ``size``.
        """
        assert self._client is not None
        resp = await self._client.file.list_path(path=path, include_size=True, show_hidden=True)
        data = resp.data
        if not data or not data.files:
            return []
        infos: list[FileInfo] = []
        for f in data.files:
            info: FileInfo = {"path": f.path}
            if f.size is not None:
                info["size"] = f.size
            infos.append(info)
        return infos

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with optional line-based pagination.

        Lines are 1-indexed in the returned text.  Each returned line is
        prefixed with its absolute line number followed by a colon and space.

        Args:
            file_path: Absolute path to the file.
            offset: Zero-based line index to start from (default: 0).
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            Formatted string with line numbers, or an error message prefixed
            with ``Error:``.
        """
        assert self._client is not None
        try:
            resp = await self._client.file.read_file(file=file_path)
            data = resp.data
            content = (data.content or "") if data else ""
        except Exception as exc:
            return f"Error: {exc}"
        lines = content.splitlines(keepends=True)
        page = lines[offset : offset + limit]
        return "".join(f"{offset + i + 1}: {line}" for i, line in enumerate(page))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write content to a new file; returns an error if the file already exists.

        Args:
            file_path: Absolute destination path.
            content: Text content to write.

        Returns:
            ``WriteResult`` with ``path`` on success or ``error`` on failure.
        """
        assert self._client is not None
        check = await self._run_bash({"command": f"test -e {shlex.quote(file_path)} && echo EXISTS || echo ABSENT"})
        if "EXISTS" in check.output:
            return WriteResult(error=f"File already exists: {file_path}")
        try:
            await self._client.file.write_file(file=file_path, content=content)
            return WriteResult(path=file_path)
        except Exception as exc:
            return WriteResult(error=str(exc))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string`` in an existing file.

        Reads the file into memory, performs the Python string replacement,
        then writes it back — giving precise control over occurrence count.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact text to search for.
            new_string: Replacement text.
            replace_all: When ``True`` replace all occurrences; when ``False``
                (default) replace only the first occurrence.

        Returns:
            ``EditResult`` with ``path`` and ``occurrences`` on success, or
            ``error`` when the string is not found or the file cannot be read.
        """
        assert self._client is not None
        try:
            resp = await self._client.file.read_file(file=file_path)
            data = resp.data
            original = (data.content or "") if data else ""
        except Exception as exc:
            return EditResult(error=f"Cannot read {file_path}: {exc}")

        count = original.count(old_string)
        if count == 0:
            return EditResult(error=f"String not found in {file_path}")

        if replace_all:
            updated = original.replace(old_string, new_string)
            occurrences = count
        else:
            updated = original.replace(old_string, new_string, 1)
            occurrences = 1

        try:
            await self._client.file.write_file(file=file_path, content=updated)
        except Exception as exc:
            return EditResult(error=f"Cannot write {file_path}: {exc}")

        return EditResult(path=file_path, occurrences=occurrences)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern in files using ``grep``.

        Args:
            pattern: Literal string to search for (exact substring match).
            path: Directory to search in; defaults to ``work_dir``.
            glob: Optional filename glob pattern to restrict the search,
                e.g. ``*.py``.

        Returns:
            List of ``GrepMatch`` dicts on success, or an error string on
            grep failure (exit code > 1).
        """
        search_path = path or self.config.work_dir
        cmd = f"grep -rna {shlex.quote(pattern)} {shlex.quote(search_path)} 2>/dev/null"
        if glob:
            cmd += f" --include={shlex.quote(glob)}"
        result = await self._run_bash({"command": cmd})
        # grep exits 0 (matches found), 1 (no matches), 2+ (errors, e.g. permission denied)
        # With 2>/dev/null, permission errors are suppressed; exit 2 means only unreadable files.
        # We still parse whatever stdout was produced before returning an error.
        matches: list[GrepMatch] = []
        for line in result.output.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    matches.append(GrepMatch(path=parts[0], line=int(parts[1]), text=parts[2]))
                except ValueError:
                    pass
        if result.exit_code > 1 and not matches and result.output.strip():
            return f"grep error: {result.output.strip()}"
        return matches

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern.

        Uses Python's ``glob.glob`` inside the sandbox (via a one-liner bash
        command) so that ``**`` recursive patterns are fully supported.

        Args:
            pattern: Glob pattern with wildcards (``*``, ``**``, ``?``,
                ``[...]``).
            path: Base directory to search from (default: ``/``).

        Returns:
            List of ``FileInfo`` dicts with the matched absolute paths.
        """
        py_cmd = (
            "import glob, os, sys; "
            f"results = glob.glob({pattern!r}, root_dir={path!r}, recursive=True); "
            f"[print(os.path.join({path!r}, r)) for r in sorted(results)]"
        )
        cmd = f"python3 -c {shlex.quote(py_cmd)}"
        result = await self._run_bash({"command": cmd})
        infos: list[FileInfo] = []
        for line in result.output.splitlines():
            line = line.strip()
            if line:
                infos.append(FileInfo(path=line))
        return infos

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files into the sandbox.

        Args:
            files: List of ``(path, content)`` tuples where content is UTF-8
                encoded bytes.

        Returns:
            List of ``FileUploadResponse`` objects in the same order as input.
            Non-UTF-8 content or write failures yield a ``permission_denied``
            error code.
        """
        assert self._client is not None
        responses: list[FileUploadResponse] = []
        for file_path, content in files:
            try:
                text = content.decode("utf-8")
                await self._client.file.write_file(file=file_path, content=text)
                responses.append(FileUploadResponse(path=file_path))
            except Exception as exc:
                logger.warning(f"upload_files failed for {file_path}: {exc}")
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Args:
            paths: Absolute file paths to download.

        Returns:
            List of ``FileDownloadResponse`` objects in the same order as
            input.  Missing or unreadable files yield a ``file_not_found``
            error code.
        """
        assert self._client is not None
        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            try:
                resp = await self._client.file.read_file(file=file_path)
                data = resp.data
                content = ((data.content or "").encode("utf-8")) if data else b""
                responses.append(FileDownloadResponse(path=file_path, content=content))
            except Exception as exc:
                logger.warning(f"download_files failed for {file_path}: {exc}")
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
        return responses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _wait_healthy(self) -> None:
        """Poll the sandbox HTTP API until it responds or startup_timeout elapses."""
        import httpx  # noqa: PLC0415

        deadline = asyncio.get_event_loop().time() + self.config.startup_timeout
        # trust_env=False: bypass http_proxy env vars that would route localhost through a corporate proxy
        async with httpx.AsyncClient(trust_env=False) as http:
            while True:
                try:
                    # Use /v1/shell/sessions — a lightweight read endpoint that returns 200 when the API is ready
                    resp = await http.get(f"{self.base_url}/v1/shell/sessions", timeout=2.0)
                    if resp.status_code < 500:
                        return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException, httpx.RemoteProtocolError):
                    pass
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Sandbox did not become healthy within {self.config.startup_timeout}s")
                await asyncio.sleep(1.0)
