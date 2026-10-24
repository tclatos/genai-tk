"""Python execution in a dedicated sandbox with an authenticated worker channel.

The container is the security boundary. An explicitly supplied backend belongs
to the caller and must not be shared between mutually untrusted sessions.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import secrets
import shlex
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from genai_tk.agents.sandbox.config import get_docker_aio_settings
from genai_tk.agents.sandbox.factory import SandboxBackendFactory
from genai_tk.agents.tools.python_executor.executor import DEFAULT_MAX_LEN_OUTPUT, MAX_EXECUTION_TIME_SECONDS
from genai_tk.agents.tools.python_executor.models import CodeOutput
from genai_tk.agents.tools.tool_specs import ClassToolSpec, FactoryToolSpec, FunctionToolSpec, UnifiedToolSpec

MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4_194_304
_JSON_VALUE = TypeAdapter(JsonValue)
_Result = TypeVar("_Result")


class WorkerReply(BaseModel):
    """Validated response from the untrusted sandbox worker."""

    kind: Literal["ready", "result", "tool_call"]
    output: JsonValue = None
    logs: str = ""
    error: str | None = None
    fatal: bool = False
    is_final_answer: bool = False
    request_id: str | None = None
    tool_name: str | None = None
    args: list[JsonValue] = Field(default_factory=list)
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class WorkerSettings(BaseModel):
    """Immutable limits supplied when starting a sandbox worker."""

    token: str
    directory: str
    timeout_seconds: float = Field(gt=0, le=3600)
    max_output_chars: int = Field(gt=0, le=1_000_000)
    memory_limit_mb: int = Field(gt=0)


def _run_sync(operation: Coroutine[Any, Any, _Result]) -> _Result:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation)
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.run(operation)


def _require_isolated_backend(backend: object) -> None:
    """Reject known host execution backends, including composite wrappers."""
    visited: set[int] = set()
    while backend is not None:
        if id(backend) in visited:
            raise ValueError("A cyclic backend cannot establish a sandbox execution boundary.")
        visited.add(id(backend))
        ancestry = {(base.__module__, base.__name__) for base in type(backend).__mro__}
        if ancestry.intersection(
            {
                ("deepagents.backends.local_shell", "LocalShellBackend"),
                ("deepagents.backends.filesystem", "FilesystemBackend"),
            }
        ):
            raise ValueError("Host shell backends cannot provide isolation; select local execution explicitly.")
        if ("deepagents.backends.composite", "CompositeBackend") not in ancestry:
            return
        backend = backend.default


class SandboxedPythonExecutor:
    """Run stateful Python in a dedicated container, without host fallback."""

    def __init__(
        self,
        backend: Any | None = None,
        additional_authorized_imports: list[str] | None = None,
        max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
        additional_functions: dict[str, Callable[..., Any]] | None = None,
        tools: dict[str, Any] | list[Any] | None = None,
        timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
        initial_state: dict[str, Any] | None = None,
        fallback_to_local: bool = False,
        memory_limit_mb: int = 1024,
    ) -> None:
        if fallback_to_local:
            raise ValueError("Host fallback is disabled. Select the local executor explicitly for trusted code.")
        if backend is not None:
            _require_isolated_backend(backend)
        self.backend = backend
        self._owns_backend = backend is None
        self._backend_started = False
        self._closed = False
        self._cleanup_complete = False
        self._cleanup_error: str | None = None
        self._worker_ready = False
        self._port: int | None = None
        self._lock = asyncio.Lock()
        self._directory = f"/tmp/genai_tk_exec_{secrets.token_hex(16)}"
        self._settings = WorkerSettings(
            token=secrets.token_urlsafe(32),
            directory=self._directory,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else MAX_EXECUTION_TIME_SECONDS,
            max_output_chars=max_print_outputs_length,
            memory_limit_mb=memory_limit_mb,
        )
        self.timeout_seconds = self._settings.timeout_seconds
        self.max_print_outputs_length = max_print_outputs_length
        # Import restrictions apply to the explicit local AST interpreter.
        # Sandbox Python has its container's normal capabilities.
        self.additional_authorized_imports = additional_authorized_imports or []
        self.additional_functions = additional_functions or {}
        self.static_tools: dict[str, BaseTool | Callable[..., object]] = {}
        self._pending_variables = self._validate_variables(initial_state or {})
        self._configuration_dirty = True
        self._init_tools(tools)

    @staticmethod
    def _validate_variables(variables: dict[str, object]) -> dict[str, JsonValue]:
        return {name: _JSON_VALUE.validate_python(value) for name, value in variables.items()}

    def _init_tools(self, tools: dict[str, Any] | list[Any] | None) -> None:
        registered: dict[str, BaseTool | Callable[..., object]] = {}
        items = list(tools.items()) if isinstance(tools, dict) else [(None, item) for item in (tools or [])]
        for alias, item in items:
            if isinstance(item, BaseTool):
                registered[alias or item.name] = item
            elif isinstance(item, (str, dict, UnifiedToolSpec, ClassToolSpec, FactoryToolSpec, FunctionToolSpec)):
                from genai_tk.agents.tools.langchain.shared_config_loader import process_langchain_tools_from_config

                for resolved in process_langchain_tools_from_config([item]):
                    if isinstance(resolved, BaseTool):
                        registered[alias or resolved.name] = resolved
            elif callable(item):
                registered[alias or item.__name__] = item
            else:
                raise TypeError("Executor tools must be tool instances, callables, or configured tool factories.")
        registered.update(self.additional_functions)
        if any(not name.isidentifier() or name.startswith("_") or name == "final_answer" for name in registered):
            raise ValueError("Tool names must be public Python identifiers other than final_answer.")
        self.static_tools = registered

    def send_tools(self, tools: dict[str, Any] | list[Any]) -> None:
        """Replace tool capabilities; old aliases cannot invoke revoked tools."""
        self._require_open()
        self.additional_functions = {}
        self._init_tools(tools)
        self._configuration_dirty = True

    def send_variables(self, variables: dict[str, Any]) -> None:
        """Inject JSON-compatible values into this session."""
        self._require_open()
        self._pending_variables.update(self._validate_variables(variables))
        self._configuration_dirty = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Executor is closed or invalidated; create a new executor and sandbox.")

    async def _get_backend(self) -> Any:
        if self.backend is None:
            config = get_docker_aio_settings().model_copy(update={"env_vars": {}, "volumes": []})
            self.backend = SandboxBackendFactory.create("docker", config=config, stop_server_on_close=False)
        _require_isolated_backend(self.backend)
        if not self._backend_started:
            if hasattr(self.backend, "start") and not getattr(self.backend, "_sandbox", None):
                await self.backend.start()
            self._backend_started = True
        return self.backend

    async def _command(self, command: str, timeout: float = 10) -> str:
        result = await asyncio.wait_for(self.backend.aexecute(command, timeout=max(1, int(timeout))), timeout + 1)
        if getattr(result, "exit_code", 0) != 0:
            raise RuntimeError("Sandbox worker command failed.")
        output = result.output
        if len(output.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Sandbox response exceeds the configured size limit.")
        return output

    async def _write_file(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await self._command(f"umask 077; : > {shlex.quote(path)}")
        # Stay below per-argument shell limits even for a valid 1 MiB request.
        for offset in range(0, len(encoded), 65536):
            chunk = encoded[offset : offset + 65536]
            await self._command(f"printf %s {shlex.quote(chunk)} | base64 -d >> {shlex.quote(path)}")

    async def _request(self, payload: dict[str, object], *, timeout: float | None = None) -> WorkerReply:
        body = json.dumps(payload, allow_nan=False)
        if len(body.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("Sandbox request exceeds the configured size limit.")
        request_path = f"{self._directory}/request_{secrets.token_hex(16)}.json"
        await self._write_file(request_path, body)
        request_timeout = timeout if timeout is not None else self.timeout_seconds + 2
        try:
            output = await self._command(
                f"curl --silent --show-error --fail --max-time {request_timeout:.3f} "
                f"--config {shlex.quote(self._directory + '/curl.conf')} "
                f"--data-binary @{shlex.quote(request_path)} http://127.0.0.1:{self._port}/",
                request_timeout,
            )
            return WorkerReply.model_validate_json(output)
        finally:
            await self._command(f"rm -f -- {shlex.quote(request_path)}")

    async def _ensure_worker(self) -> None:
        if not self._worker_ready:
            await self._get_backend()
            await self._command(f"mkdir -m 700 -- {shlex.quote(self._directory)}")
            worker_path = f"{self._directory}/worker.py"
            await self._write_file(worker_path, Path(__file__).with_name("sandbox_worker.py").read_text())
            await self._write_file(f"{self._directory}/settings.json", self._settings.model_dump_json())
            await self._write_file(
                f"{self._directory}/curl.conf",
                f'header = "Authorization: Bearer {self._settings.token}"\nheader = "Content-Type: application/json"\n',
            )
            await self._command(
                "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                f"nohup python3 -I -u {shlex.quote(worker_path)} "
                f"--config {shlex.quote(self._directory + '/settings.json')} "
                f"> {shlex.quote(self._directory + '/worker.log')} 2>&1 < /dev/null &"
            )
            for _ in range(50):
                output = await self._command(f"cat {shlex.quote(self._directory + '/port')} 2>/dev/null || true")
                if output.strip().isdigit():
                    self._port = int(output.strip())
                    if not 1 <= self._port <= 65535:
                        raise ValueError("Worker returned an invalid port.")
                    break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("Sandbox worker did not become ready.")
            self._worker_ready = True
        if self._configuration_dirty:
            reply = await self._request(
                {"action": "configure", "tools": list(self.static_tools), "variables": self._pending_variables}
            )
            if reply.kind != "ready":
                raise RuntimeError(reply.error or "Worker rejected session configuration.")
            self._pending_variables = {}
            self._configuration_dirty = False

    async def _invoke_tool(self, reply: WorkerReply) -> dict[str, object]:
        tool = self.static_tools.get(reply.tool_name or "")
        if tool is None:
            return {"error": "Tool is not authorized in this session."}
        try:
            if isinstance(tool, BaseTool):
                tool_input = reply.kwargs or (reply.args[0] if len(reply.args) == 1 else {})
                value = await tool.ainvoke(tool_input)
            elif inspect.iscoroutinefunction(tool):
                value = await tool(*reply.args, **reply.kwargs)
            else:
                value = await asyncio.to_thread(tool, *reply.args, **reply.kwargs)
                if inspect.isawaitable(value):
                    value = await value
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="json")
            elif hasattr(value, "content"):
                value = value.content
            return {"result": _JSON_VALUE.validate_python(value)}
        except Exception as exc:
            return {"error": f"Host tool failed: {type(exc).__name__}: {exc}"}

    async def aexecute_code(self, code: str) -> CodeOutput:
        """Execute once; never retry code or tool calls after infrastructure errors."""
        async with self._lock:
            try:
                self._require_open()
                await self._ensure_worker()
                deadline = time.monotonic() + self.timeout_seconds
                reply = await self._request({"action": "execute", "code": code})
                while reply.kind == "tool_call":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Python execution exceeded its deadline.")
                    tool_result = await asyncio.wait_for(self._invoke_tool(reply), remaining)
                    reply = await self._request(
                        {"action": "tool_result", "request_id": reply.request_id, **tool_result},
                        timeout=max(0.1, deadline - time.monotonic()) + 2,
                    )
                if reply.kind != "result":
                    raise RuntimeError("Unexpected sandbox worker response.")
                if reply.fatal:
                    cleanup_error = await self._retire()
                    if cleanup_error:
                        reply.error = f"{reply.error or 'Worker failed.'} {cleanup_error}"
                return CodeOutput(
                    output=reply.output, logs=reply.logs, error=reply.error, is_final_answer=reply.is_final_answer
                )
            except asyncio.CancelledError as exc:
                cleanup_error = await self._retire()
                if cleanup_error:
                    exc.add_note(cleanup_error)
                raise
            except Exception as exc:
                cleanup_error = await self._retire()
                logger.warning("Sandbox Python execution failed closed: {}", type(exc).__name__)
                error = f"Sandbox execution failed; executor invalidated: {type(exc).__name__}: {exc}"
                return CodeOutput(error=f"{error} {cleanup_error}" if cleanup_error else error)

    async def _retire(self, *, failed: bool = True) -> str | None:
        self._closed = True
        if self._cleanup_complete:
            return self._cleanup_error
        self._cleanup_error = None
        if self._owns_backend and self.backend is not None:
            stop = getattr(self.backend, "aclose", None) or getattr(self.backend, "stop", None)
            try:
                if stop is None:
                    raise RuntimeError("Backend has no shutdown method.")
                await asyncio.wait_for(stop(), 10)
                self._cleanup_complete = True
                self._worker_ready = False
            except Exception as exc:
                self._cleanup_error = (
                    f"Sandbox termination could not be confirmed ({type(exc).__name__}); "
                    "retry aclose() or have the operator retire the container."
                )
                logger.error(self._cleanup_error)
        else:
            try:
                if self._worker_ready:
                    await self._request({"action": "shutdown"}, timeout=2)
                    await self._command(f"rm -rf -- {shlex.quote(self._directory)}")
                    self._worker_ready = False
                self._cleanup_complete = True
            except Exception as exc:
                self._cleanup_error = (
                    f"Worker termination could not be confirmed ({type(exc).__name__}); "
                    "retry aclose() and retire the caller-owned sandbox."
                )
            if failed and self._backend_started:
                ownership_warning = (
                    "The caller must retire its backend to guarantee all sandbox processes have stopped."
                )
                self._cleanup_error = (
                    f"{self._cleanup_error} {ownership_warning}" if self._cleanup_error else ownership_warning
                )
        return self._cleanup_error

    async def _areset(self) -> None:
        async with self._lock:
            self._require_open()
            try:
                self._pending_variables = {}
                if self._worker_ready:
                    reply = await self._request({"action": "reset"})
                    if reply.kind != "ready":
                        raise RuntimeError("Worker rejected the reset request.")
                self._configuration_dirty = True
            except Exception as exc:
                cleanup_error = await self._retire()
                message = "Sandbox reset failed; executor invalidated."
                raise RuntimeError(f"{message} {cleanup_error}" if cleanup_error else message) from exc

    def reset(self) -> None:
        """Clear all variables and functions while keeping configured capabilities."""
        _run_sync(self._areset())

    def __call__(self, code: str) -> CodeOutput:
        return _run_sync(self.aexecute_code(code))

    async def aclose(self) -> None:
        """Close the worker and its owned backend; borrowed backends remain caller-owned."""
        async with self._lock:
            if self._cleanup_complete:
                return
            cleanup_error = await self._retire(failed=False)
            if cleanup_error:
                raise RuntimeError(cleanup_error)

    def close(self) -> None:
        _run_sync(self.aclose())

    async def __aenter__(self) -> SandboxedPythonExecutor:
        self._require_open()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


DockerPythonExecutor = SandboxedPythonExecutor
