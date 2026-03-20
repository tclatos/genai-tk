"""AioSandboxBackend — deepagents BackendProtocol backed by OpenSandbox + agent-infra/sandbox.

Uses an OpenSandbox server to manage the container lifecycle (dynamic port
allocation, concurrent sandboxes) and ``agent_sandbox.AsyncSandbox`` as the
HTTP client for shell/file operations.

Prerequisites: ``uv add opensandbox agent-sandbox`` and ``opensandbox-server`` running.

Example:
    ```python
    from genai_tk.agents.sandbox import AioSandboxBackend

    async with AioSandboxBackend() as backend:
        resp = await backend.aexecute("echo hello")
        print(resp.output)
    ```
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from genai_tk.agents.sandbox.models import DockerAioSettings

if TYPE_CHECKING:
    from agent_sandbox import AsyncSandbox


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


class AioSandboxBackend(SandboxBackendProtocol, BaseModel):
    """deepagents ``SandboxBackendProtocol`` backed by OpenSandbox + ``agent_sandbox.AsyncSandbox``.

    Lifecycle is managed by an OpenSandbox server; the ``a*`` protocol methods
    are async-native and communicate via the agent-sandbox HTTP client.
    """

    config: DockerAioSettings = Field(default_factory=DockerAioSettings)

    _client: AsyncSandbox | None = None
    _sandbox: object | None = None
    _server_proc: object | None = None
    _base_url: str = PrivateAttr(default="")
    _instance_id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex[:12])
    _extra_volumes: list = PrivateAttr(default_factory=list)  # runtime-added VolumeMountConfig items

    model_config = {"arbitrary_types_allowed": True}

    @property
    def id(self) -> str:
        """Unique sandbox identifier."""
        return getattr(self._sandbox, "id", self._instance_id)

    def add_volume(self, host_path: str, container_path: str, *, read_only: bool = False) -> None:
        """Register an additional bind-mount to include when the sandbox starts.

        Must be called **before** ``start()``.
        """
        from genai_tk.agents.sandbox.models import VolumeMountConfig  # noqa: PLC0415

        self._extra_volumes.append(
            VolumeMountConfig(host_path=host_path, container_path=container_path, read_only=read_only)
        )

    def _build_volumes(self) -> list:
        """Convert config + runtime volume mounts into opensandbox ``Volume`` objects."""
        from opensandbox.models.sandboxes import Host, Volume  # noqa: PLC0415

        from genai_tk.agents.sandbox.models import VolumeMountConfig  # noqa: PLC0415

        all_mounts: list[VolumeMountConfig] = list(self.config.volumes) + list(self._extra_volumes)
        if not all_mounts:
            return []

        volumes: list[Volume] = []
        for i, m in enumerate(all_mounts):
            volumes.append(
                Volume(
                    name=f"vol-{i}",
                    host=Host(path=m.host_path),
                    mount_path=m.container_path,
                    read_only=m.read_only,
                )
            )
            logger.debug(f"Volume mount: {m.host_path} → {m.container_path} (ro={m.read_only})")
        return volumes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the sandbox via OpenSandbox and wait until healthy.

        Auto-starts ``opensandbox-server`` if it is not already reachable.
        """
        from datetime import timedelta  # noqa: PLC0415
        from urllib.parse import urlparse  # noqa: PLC0415

        try:
            from opensandbox import Sandbox  # noqa: PLC0415
            from opensandbox.config import ConnectionConfig  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("opensandbox is required: uv add opensandbox") from exc
        try:
            from agent_sandbox import AsyncSandbox  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("agent-sandbox is required: uv add agent-sandbox") from exc
        import httpx  # noqa: PLC0415

        server_url = self.config.opensandbox_server_url
        self._server_proc = await self._ensure_server(server_url)
        parsed = urlparse(server_url)
        conn_config = ConnectionConfig(domain=parsed.netloc or server_url, protocol=parsed.scheme or "http")
        startup_timeout = self.config.startup_timeout

        async def _health_check(sbx: object) -> bool:
            try:
                ep = await sbx.get_endpoint(8080)  # type: ignore[attr-defined]
            except Exception:
                return False
            url = f"http://{ep.endpoint}/v1/shell/sessions"
            deadline = asyncio.get_event_loop().time() + startup_timeout
            async with httpx.AsyncClient(trust_env=False) as hc:
                while True:
                    try:
                        if (await hc.get(url, timeout=2.0)).status_code < 500:
                            return True
                    except Exception:
                        pass
                    if asyncio.get_event_loop().time() > deadline:
                        return False
                    await asyncio.sleep(1.0)

        # Build volume mounts from config + runtime additions
        volumes = self._build_volumes()

        # Merge environment vars: inject browser flags for SSL compatibility and
        # anti-bot stealth.  The sandbox container's entrypoint appends
        # BROWSER_EXTRA_ARGS to Chromium's command line and uses BROWSER_USER_AGENT
        # for the --user-agent flag.
        env = dict(self.config.env_vars)
        if "BROWSER_EXTRA_ARGS" not in env:
            env["BROWSER_EXTRA_ARGS"] = "--ignore-certificate-errors --disable-blink-features=AutomationControlled"

        logger.debug(f"Starting AIO sandbox via {server_url}")
        create_kwargs: dict[str, Any] = {
            "timeout": timedelta(seconds=startup_timeout),
            "entrypoint": self.config.entrypoint,
            "env": env,
            "connection_config": conn_config,
            "health_check": _health_check,
        }
        if volumes:
            create_kwargs["volumes"] = volumes
            logger.info(f"Mounting {len(volumes)} volume(s) into sandbox")
        self._sandbox = await Sandbox.create(self.config.image, **create_kwargs)
        endpoint = await self._sandbox.get_endpoint(8080)  # type: ignore[attr-defined]
        self._base_url = f"http://{endpoint.endpoint}"
        self._client = AsyncSandbox(
            base_url=self._base_url, httpx_client=httpx.AsyncClient(trust_env=False, timeout=120)
        )
        logger.info(f"AioSandbox ready at {self._base_url}")
        logger.info(f"VNC (visual debug): {self._base_url}/vnc/index.html?autoconnect=true")

    async def stop(self) -> None:
        """Kill the sandbox and stop the server if we started it."""
        if self._client is not None:
            try:
                await self._client.httpx_client.aclose()  # type: ignore[union-attr]
            except Exception:
                pass
        self._client = None
        if self._sandbox is not None:
            sbx, self._sandbox = self._sandbox, None
            try:
                await sbx.kill()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug(f"sandbox kill (non-critical): {exc}")
            logger.info("AioSandbox stopped")
        if self._server_proc is not None:
            proc, self._server_proc = self._server_proc, None
            proc.terminate()  # type: ignore[attr-defined]
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)  # type: ignore[attr-defined]
            except asyncio.TimeoutError:
                proc.kill()  # type: ignore[attr-defined]
            # Clean up the PID file we wrote during _ensure_server
            pid_file = Path.home() / ".cache" / "genai-tk" / "opensandbox-server.pid"
            pid_file.unlink(missing_ok=True)
            logger.info("opensandbox-server stopped")

    def detach(self) -> None:
        """Release all references without killing processes or containers.

        Used by ``--keep-sandbox`` to prevent asyncio’s subprocess transport
        ``__del__`` from sending SIGKILL to the opensandbox-server on exit.
        The server PID file is preserved so ``cli sandbox stop`` still works.
        """
        # Close the httpx client — we don't need it
        if self._client is not None:
            try:
                import asyncio as _aio

                loop = _aio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.httpx_client.aclose())  # type: ignore[union-attr]
            except Exception:
                pass
        self._client = None

        # Detach the asyncio subprocess transport so __del__ won't kill the server.
        # The actual OS process survives because we used start_new_session=True.
        if self._server_proc is not None:
            transport = getattr(self._server_proc, "_transport", None)
            if transport is not None:
                transport._closed = True  # prevent __del__ -> close() -> kill()
            self._server_proc = None

        # Release the sandbox reference without killing the container.
        self._sandbox = None
        self._connected = False  # type: ignore[attr-defined]
        logger.info("AioSandbox detached — server and container left running")

    async def _ensure_server(self, server_url: str) -> object | None:
        """Return ``None`` if the server is already up, otherwise start it and return the process.

        The server is started in a new session (``start_new_session=True``) so it
        survives the parent process exiting — important for ``--keep-sandbox``.
        A PID file is written so ``cli sandbox stop`` can find and terminate it.
        """
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        check_url = f"{server_url}/v1/sandboxes"
        try:
            async with httpx.AsyncClient(trust_env=False) as hc:
                await hc.get(check_url, timeout=2.0)
            return None  # already running
        except Exception:
            pass

        # Resolve the server binary: prefer the one next to this Python interpreter
        # (i.e. same virtualenv), fall back to PATH.
        venv_bin = Path(sys.executable).parent / "opensandbox-server"
        if venv_bin.is_file():
            server_cmd = str(venv_bin)
        else:
            server_cmd = shutil.which("opensandbox-server") or "opensandbox-server"

        logger.info(f"opensandbox-server not reachable at {server_url} — starting it")
        try:
            proc = await asyncio.create_subprocess_exec(
                server_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from parent so the server survives process exit
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise RuntimeError(
                "opensandbox-server not found. "
                "Install: uv add opensandbox-server && opensandbox-server init-config ~/.sandbox.toml --example docker"
            ) from exc

        deadline = asyncio.get_event_loop().time() + 15
        async with httpx.AsyncClient(trust_env=False) as hc:
            while True:
                try:
                    await hc.get(check_url, timeout=2.0)
                    logger.info("opensandbox-server ready")
                    # Write PID file so `cli sandbox stop` can find the server
                    pid_file = Path.home() / ".cache" / "genai-tk" / "opensandbox-server.pid"
                    pid_file.parent.mkdir(parents=True, exist_ok=True)
                    pid_file.write_text(str(proc.pid))
                    return proc
                except Exception:
                    pass
                if asyncio.get_event_loop().time() > deadline:
                    proc.terminate()
                    raise RuntimeError(
                        "opensandbox-server did not become healthy within 15s. "
                        "Check config: opensandbox-server init-config ~/.sandbox.toml --example docker"
                    )
                await asyncio.sleep(0.5)

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
    # Tool implementations
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
            if f.is_directory:
                info["is_dir"] = True
            if f.size is not None:
                info["size"] = f.size
            infos.append(info)
        return infos

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with optional line-based pagination.

        Lines are 1-indexed in the returned text.

        Args:
            file_path: Absolute path to the file.
            offset: Zero-based line index to start from (default: 0).
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            Formatted string with line numbers, or an error message prefixed with ``Error:``.
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

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact text to search for.
            new_string: Replacement text.
            replace_all: Replace all occurrences when ``True`` (default: first only).

        Returns:
            ``EditResult`` with ``path`` and ``occurrences`` on success, or ``error``.
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
            glob: Optional filename glob to restrict the search, e.g. ``*.py``.

        Returns:
            List of ``GrepMatch`` dicts on success, or an error string on grep failure.
        """
        search_path = path or self.config.work_dir
        cmd = f"grep -rna {shlex.quote(pattern)} {shlex.quote(search_path)} 2>/dev/null"
        if glob:
            cmd += f" --include={shlex.quote(glob)}"
        result = await self._run_bash({"command": cmd})
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

        Args:
            pattern: Glob pattern with wildcards (``*``, ``**``, ``?``, ``[...]``).
            path: Base directory to search from (default: ``/``).

        Returns:
            List of ``FileInfo`` dicts with matched absolute paths.
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
            files: List of ``(path, content)`` tuples where content is UTF-8 bytes.

        Returns:
            List of ``FileUploadResponse`` objects in the same order as input.
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
            List of ``FileDownloadResponse`` in the same order as input.
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
