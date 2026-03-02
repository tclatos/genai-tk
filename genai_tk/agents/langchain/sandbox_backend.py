"""LangChain deepagents BackendProtocol backed by agent-infra/sandbox.

Uses ``agent_sandbox.AsyncSandbox`` as an HTTP client against a running Docker
container (``ghcr.io/agent-infra/sandbox``).  This module manages the full
container lifecycle: starts the container on ``start()``, polls until the HTTP
API is healthy, then stops + removes the container on ``stop()``.

The tool implementations mirror Deer-flow's own ``AioSandbox`` class:
``shell.exec_command`` for bash/ls, ``file.*`` endpoints for file I/O.

Example:
```python
from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

async with AioSandboxBackend() as backend:
    result = await backend.execute_tool("bash", {"command": "echo hello"})
    print(result.output)
```
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

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


class AioSandboxBackend(BaseModel):
    """deepagents BackendProtocol backed by ``agent_sandbox.AsyncSandbox``.

    Manages the Docker container lifecycle: starts on ``__aenter__``, exposes
    ``bash``, ``ls``, ``read_file``, ``write_file``, ``str_replace`` tools,
    and stops the container on ``__aexit__``.
    """

    config: AioSandboxBackendConfig = Field(default_factory=AioSandboxBackendConfig)

    _container_id: str | None = None
    _client: AsyncSandbox | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.host_port}"

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
