"""AioSandboxBackend — deepagents BackendProtocol backed by OpenSandbox.

Uses an OpenSandbox server to manage the container lifecycle (dynamic port
allocation, concurrent sandboxes) and the native ``opensandbox.Sandbox``
command/filesystem services (the in-container ``execd`` daemon) for shell
and file operations.

The agent speaks to the execd daemon (``opensandbox/execd``) that the
``opensandbox`` bootstrap injects into the sandbox container — NOT to the
OpenSandbox Lifecycle API that the same bootstrap exposes on port 8080.
Prerequisites: ``uv add opensandbox`` and ``opensandbox-server`` running.

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

from genai_tk.config_mgmt.features import require_feature

require_feature("harnessing", context="genai_tk.agents.sandbox.aio_backend")

from deepagents.backends.protocol import (  # noqa: E402
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from loguru import logger  # noqa: E402
from pydantic import BaseModel, Field, PrivateAttr  # noqa: E402

from genai_tk.agents.sandbox.models import DockerAioSettings  # noqa: E402

if TYPE_CHECKING:
    from opensandbox import Sandbox


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
    """deepagents ``SandboxBackendProtocol`` backed by OpenSandbox.

    Lifecycle is managed by an OpenSandbox server; the ``a*`` protocol methods
    are async-native and communicate with the in-container execd daemon through
    the ``opensandbox`` SDK's command and filesystem services.
    """

    config: DockerAioSettings = Field(default_factory=DockerAioSettings)

    _sandbox: Sandbox | None = None
    _server_proc: object | None = None
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

    def _build_browser_env(self) -> dict[str, str]:
        """Build browser-related environment variables for the sandbox container."""
        from genai_tk.agents.tools.sandbox_browser.factory import _load_browser_config  # noqa: PLC0415

        browser_config = _load_browser_config()
        locale_tag = browser_config.locale
        locale_env = f"{locale_tag.replace('-', '_')}.UTF-8"
        env = dict(self.config.env_vars)

        # The sandbox image bakes in a desktop-Mac Chrome UA by default via
        # BROWSER_USER_AGENT. That value can drift from the actual Chromium
        # version and navigator.userAgentData brands, which is detectable.
        # Clear it unless the caller explicitly opted into a custom UA.
        env.setdefault("BROWSER_USER_AGENT", "")
        env.setdefault("TZ", browser_config.timezone_id)
        env.setdefault("LANG", locale_env)
        env.setdefault("LC_ALL", locale_env)
        env.setdefault("LANGUAGE", f"{locale_tag.replace('-', '_')}:{locale_tag.split('-')[0]}")

        # The sandbox entrypoint appends BROWSER_EXTRA_ARGS to Chromium's
        # command line. Keep our generic webdriver suppression flag, but do
        # not inject TLS-altering flags such as --ignore-certificate-errors.
        browser_extra_args = env.get("BROWSER_EXTRA_ARGS", "").strip()
        if "--lang=" not in browser_extra_args:
            browser_extra_args = f"{browser_extra_args} --lang={browser_config.locale}".strip()
        if "--time-zone-for-testing=" not in browser_extra_args:
            browser_extra_args = f"{browser_extra_args} --time-zone-for-testing={browser_config.timezone_id}".strip()
        if "--disable-blink-features=AutomationControlled" not in browser_extra_args:
            browser_extra_args = f"{browser_extra_args} --disable-blink-features=AutomationControlled".strip()
        env["BROWSER_EXTRA_ARGS"] = browser_extra_args

        return env

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the sandbox via OpenSandbox and wait until the execd is healthy.

        Auto-starts ``opensandbox-server`` if it is not already reachable.
        """
        from datetime import timedelta  # noqa: PLC0415
        from urllib.parse import urlparse  # noqa: PLC0415

        try:
            from opensandbox import Sandbox  # noqa: PLC0415
            from opensandbox.config import ConnectionConfig  # noqa: PLC0415
        except ImportError as exc:
            from genai_tk.config_mgmt.features import require_feature  # noqa: PLC0415

            require_feature("harnessing", context="AioSandboxBackend.start")
            raise RuntimeError("unexpected") from exc

        server_url = self.config.opensandbox_server_url
        self._server_proc = await self._ensure_server(server_url)
        parsed = urlparse(server_url)
        conn_config = ConnectionConfig(domain=parsed.netloc or server_url, protocol=parsed.scheme or "http")
        startup_timeout = self.config.startup_timeout

        # Build volume mounts from config + runtime additions
        volumes = self._build_volumes()

        # Merge environment vars used by the sandbox browser entrypoint.
        env = self._build_browser_env()

        logger.debug(f"Starting AIO sandbox via {server_url}")
        create_kwargs: dict[str, Any] = {
            # `timeout` is the sandbox's MAX LIFETIME (opensandbox default: 600s),
            # NOT how long to wait for startup — that's `ready_timeout` below.
            # This backend explicitly kills the sandbox itself in stop()/aclose(),
            # so pass None ("require explicit cleanup") rather than reusing
            # `startup_timeout` here: that previously self-destructed every sandbox
            # ~60s into a session regardless of activity, causing "sandbox not
            # found" connectivity failures on any turn/conversation longer than that.
            "timeout": None,
            "ready_timeout": timedelta(seconds=startup_timeout),
            "entrypoint": self.config.entrypoint,
            "env": env,
            "connection_config": conn_config,
        }
        if volumes:
            create_kwargs["volumes"] = volumes
            logger.info(f"Mounting {len(volumes)} volume(s) into sandbox")
        # Let the SDK run its default health check (ping the execd daemon).
        self._sandbox = await Sandbox.create(self.config.image, **create_kwargs)
        logger.info(f"AioSandbox ready (sandbox id={self._sandbox.id})")

    async def stop(self) -> None:
        """Kill the sandbox and stop the server if we started it."""
        if self._sandbox is not None:
            sbx, self._sandbox = self._sandbox, None
            try:
                await sbx.kill()  # type: ignore[union-attr]
            except Exception as exc:
                logger.debug(f"sandbox kill (non-critical): {exc}")
            try:
                await sbx.close()  # type: ignore[union-attr]
            except Exception as exc:
                logger.debug(f"sandbox close (non-critical): {exc}")
            logger.info("AioSandbox stopped")
        if self._server_proc is not None:
            proc, self._server_proc = self._server_proc, None
            proc.terminate()  # type: ignore[attr-defined]
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)  # type: ignore[arg-type]
            except asyncio.TimeoutError:
                proc.kill()  # type: ignore[attr-defined]
            # Clean up the PID file we wrote during _ensure_server
            pid_file = Path.home() / ".cache" / "genai-tk" / "opensandbox-server.pid"
            pid_file.unlink(missing_ok=True)
            logger.info("opensandbox-server stopped")

    def detach(self) -> None:
        """Release all references without killing processes or containers.

        Used by ``--keep-sandbox`` to prevent asyncio's subprocess transport
        ``__del__`` from sending SIGKILL to the opensandbox-server on exit.
        The server PID file is preserved so ``cli sandbox stop`` still works.
        """
        # Detach the asyncio subprocess transport so __del__ won't kill the server.
        # The actual OS process survives because we used start_new_session=True.
        if self._server_proc is not None:
            transport = getattr(self._server_proc, "_transport", None)
            if transport is not None:
                transport._closed = True  # prevent __del__ -> close() -> kill()
            self._server_proc = None

        # Release the sandbox reference without killing the container.
        self._sandbox = None
        logger.info("AioSandbox detached — server and container left running")

    async def _ensure_server(self, server_url: str) -> object | None:
        """Return ``None`` if the server is already up, otherwise start it and return the process.

        The server is started in a new session (``start_new_session=True``) so it
        survives the parent process exiting — important for ``--keep-sandbox``.
        A PID file is written so ``cli sandbox stop`` can find and terminate it.
        """
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from urllib.parse import urlparse as _urlparse  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        from genai_tk.agents.sandbox.config import write_server_config  # noqa: PLC0415

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

        # Build a minimal server config with the correct port so multiple test
        # instances (or non-default ports) don't collide with the default 8080.
        _parsed_url = _urlparse(server_url)
        _server_port = _parsed_url.port or 8080
        _tmp_cfg_path = write_server_config(_server_port)

        # OPENSANDBOX_INSECURE_SERVER=YES bypasses the interactive confirmation
        # prompt that fires when api_key is empty (non-interactive safe path).
        _server_env = {**os.environ, "OPENSANDBOX_INSECURE_SERVER": "YES", "SANDBOX_CONFIG_PATH": str(_tmp_cfg_path)}

        try:
            proc = await asyncio.create_subprocess_exec(
                server_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from parent so the server survives process exit
                env=_server_env,
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
                    if proc.returncode is None:  # only terminate if process is still alive
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
        if self._sandbox is None:
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
        assert self._sandbox is not None
        command: str = tool_input["command"]
        try:
            execution = await self._sandbox.commands.run(command)
        except Exception as exc:
            # Connectivity errors (execd unreachable — container stopped, idle-evicted,
            # or opensandbox-server restarted) otherwise propagate as raw httpx/opensandbox
            # tracebacks through agrep/als/aexecute, which don't catch exceptions like the
            # file-op methods below do. Convert to a clear, actionable tool error instead.
            error_msg = (
                f"Cannot reach the sandbox container to run this command: {exc}. "
                "The sandbox may have stopped, been idle-evicted, or opensandbox-server "
                "may have restarted — try 'cli sandbox status' or start a new turn to "
                "recreate the sandbox."
            )
            logger.warning(f"_run_bash connectivity failure: {error_msg}")
            return SandboxToolResult(tool_name="bash", output="", exit_code=1, error=error_msg)
        output = execution.text
        exit_code = execution.exit_code if execution.exit_code is not None else 0
        return SandboxToolResult(tool_name="bash", output=output, exit_code=exit_code)

    async def _run_ls(self, tool_input: dict) -> SandboxToolResult:
        path: str = tool_input.get("path", self.config.work_dir)
        result = await self._run_bash({"command": f"ls -1pA {shlex.quote(path)} 2>/dev/null"})
        return SandboxToolResult(tool_name="ls", output=result.output, exit_code=result.exit_code)

    async def _run_read_file(self, tool_input: dict) -> SandboxToolResult:
        assert self._sandbox is not None
        file_path: str = tool_input["path"]
        content = await self._sandbox.files.read_file(file_path)
        return SandboxToolResult(tool_name="read_file", output=content)

    async def _run_write_file(self, tool_input: dict) -> SandboxToolResult:
        assert self._sandbox is not None
        file_path: str = tool_input["path"]
        content: str = tool_input["content"]
        await self._sandbox.files.write_file(file_path, content)
        return SandboxToolResult(tool_name="write_file", output=f"Written: {file_path}")

    async def _run_str_replace(self, tool_input: dict) -> SandboxToolResult:
        assert self._sandbox is not None
        file_path: str = tool_input["path"]
        old_str: str = tool_input["old_str"]
        new_str: str = tool_input["new_str"]
        try:
            original = await self._sandbox.files.read_file(file_path)
        except Exception as exc:
            return SandboxToolResult(
                tool_name="str_replace",
                output="",
                exit_code=1,
                error=f"Cannot read {file_path}: {exc}",
            )
        count = original.count(old_str)
        if count == 0:
            return SandboxToolResult(
                tool_name="str_replace",
                output="",
                exit_code=1,
                error=f"String not found in {file_path}",
            )
        try:
            await self._sandbox.files.write_file(file_path, original.replace(old_str, new_str))
        except Exception as exc:
            return SandboxToolResult(
                tool_name="str_replace",
                output="",
                exit_code=1,
                error=f"Cannot write {file_path}: {exc}",
            )
        return SandboxToolResult(
            tool_name="str_replace",
            output=f"Replaced {count}x in: {file_path}",
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
        # ExecuteResponse has no `error` field — fold any connectivity error into
        # `output` (with a non-zero exit code) so it's visible to the caller/LLM
        # instead of being silently dropped.
        output = result.error or result.output
        return ExecuteResponse(output=output, exit_code=result.exit_code)

    # ------------------------------------------------------------------
    # BackendProtocol — file operations
    # ------------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        """List directory contents (direct children only).

        Args:
            path: Absolute path to the directory.

        Returns:
            ``LsResult`` with ``entries`` on success or ``error`` on failure.
        """
        result = await self._run_bash({"command": f"ls -1pA {shlex.quote(path)} 2>/dev/null"})
        if result.exit_code != 0 and not result.output:
            return LsResult(error=result.error or f"ls failed: {path}")
        infos: list[dict[str, Any]] = []
        for line in result.output.splitlines():
            name = line.strip()
            if not name:
                continue
            is_dir = name.endswith("/")
            name = name.rstrip("/")
            info: dict[str, Any] = {"path": str(Path(path) / name)}
            if is_dir:
                info["is_dir"] = True
            infos.append(info)
        return LsResult(entries=infos)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file with optional line-based pagination.

        Lines are 1-indexed in the returned text.

        Args:
            file_path: Absolute path to the file.
            offset: Zero-based line index to start from (default: 0).
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            ``ReadResult`` with ``file_data`` on success or ``error`` on failure.
        """
        assert self._sandbox is not None
        try:
            content = await self._sandbox.files.read_file(file_path)
        except Exception as exc:
            return ReadResult(error=f"Error: {exc}")
        lines = content.splitlines(keepends=True)
        page = lines[offset : offset + limit]
        formatted = "".join(f"{offset + i + 1}: {line}" for i, line in enumerate(page))
        return ReadResult(file_data={"content": formatted, "encoding": "utf-8"})

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write content to a new file; returns an error if the file already exists.

        Args:
            file_path: Absolute destination path.
            content: Text content to write.

        Returns:
            ``WriteResult`` with ``path`` on success or ``error`` on failure.
        """
        assert self._sandbox is not None
        check = await self._run_bash({"command": f"test -e {shlex.quote(file_path)} && echo EXISTS || echo ABSENT"})
        if "EXISTS" in check.output:
            return WriteResult(error=f"File already exists: {file_path}")
        try:
            await self._sandbox.files.write_file(file_path, content)
            return WriteResult(path=file_path)
        except Exception as exc:
            return WriteResult(error=str(exc))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
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
        assert self._sandbox is not None
        try:
            original = await self._sandbox.files.read_file(file_path)
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
            await self._sandbox.files.write_file(file_path, updated)
        except Exception as exc:
            return EditResult(error=f"Cannot write {file_path}: {exc}")

        return EditResult(path=file_path, occurrences=occurrences)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal text pattern in files using ``grep``.

        Args:
            pattern: Literal string to search for (exact substring match).
            path: Directory to search in; defaults to ``work_dir``.
            glob: Optional filename glob to restrict the search, e.g. ``*.py``.

        Returns:
            ``GrepResult`` with ``matches`` on success or ``error`` on grep failure.
        """
        search_path = path or self.config.work_dir
        cmd = f"grep -rna {shlex.quote(pattern)} {shlex.quote(search_path)} 2>/dev/null"
        if glob:
            cmd += f" --include={shlex.quote(glob)}"
        result = await self._run_bash({"command": cmd})
        if result.error:
            return GrepResult(error=result.error)
        matches: list[GrepMatch] = []
        for line in result.output.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    matches.append(GrepMatch(path=parts[0], line=int(parts[1]), text=parts[2]))
                except ValueError:
                    pass
        if result.exit_code > 1 and not matches and result.output.strip():
            return GrepResult(error=f"grep error: {result.output.strip()}")
        return GrepResult(matches=matches)

    async def als_info(self, path: str = "/") -> list[dict[str, Any]]:
        """Deprecated alias for ``als()`` returning directory entries."""
        result = await self.als(path)
        return result.entries or []

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Deprecated alias for ``agrep()`` returning matches list or error string."""
        result = await self.agrep(pattern, path=path, glob=glob)
        if result.error:
            return result.error
        return result.matches or []

    async def aglob_info(self, pattern: str, path: str = "/") -> list[dict[str, Any]]:
        """Deprecated alias for ``aglob()`` returning file info matches."""
        result = await self.aglob(pattern, path=path)
        return result.matches or []

    async def aglob(self, pattern: str, path: str = "/"):  # type: ignore[override]
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern with wildcards (``*``, ``**``, ``?``, ``[...]``).
            path: Base directory to search from (default: ``/``).

        Returns:
            ``GlobResult`` with ``matches`` (list of ``FileInfo``) on success.
        """
        from deepagents.backends.protocol import GlobResult  # noqa: PLC0415
        from opensandbox.models.filesystem import SearchEntry  # noqa: PLC0415

        assert self._sandbox is not None
        entries = await self._sandbox.files.search(SearchEntry(path=path, pattern=pattern))
        infos: list[dict[str, Any]] = []
        for e in entries:
            info: dict[str, Any] = {"path": e.path}
            if e.size is not None:
                info["size"] = e.size
            infos.append(info)
        return GlobResult(matches=infos)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files into the sandbox.

        Args:
            files: List of ``(path, content)`` tuples where content is UTF-8 bytes.

        Returns:
            List of ``FileUploadResponse`` objects in the same order as input.
        """
        assert self._sandbox is not None
        responses: list[FileUploadResponse] = []
        for file_path, content in files:
            try:
                text = content.decode("utf-8")
                await self._sandbox.files.write_file(file_path, text)
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
        assert self._sandbox is not None
        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            try:
                content = await self._sandbox.files.read_bytes(file_path)
                responses.append(FileDownloadResponse(path=file_path, content=content))
            except Exception as exc:
                logger.warning(f"download_files failed for {file_path}: {exc}")
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
        return responses
