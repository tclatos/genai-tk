"""Docker sandbox Python code executor.

Executes Python code inside the all-in-one Docker sandbox container (ghcr.io/agent-infra/sandbox)
with state persistence across calls, output/log capture, final_answer support for CodeAct,
and bidirectional host tool bridging.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
from genai_tk.agents.sandbox.manager import DockerSandboxManager
from genai_tk.agents.tools.python_executor.executor import (
    DEFAULT_MAX_LEN_OUTPUT,
    MAX_EXECUTION_TIME_SECONDS,
)
from genai_tk.agents.tools.python_executor.models import CodeOutput

WORKER_SCRIPT_PATH = "/tmp/genai_tk_pyworker.py"
WORKER_PORT = 9199

# Python worker daemon code executed inside the sandbox container
IN_CONTAINER_WORKER_CODE = """
import sys
import io
import json
import traceback
import ast
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

namespace = {"__name__": "__main__"}

class FinalAnswerException(BaseException):
    def __init__(self, val):
        self.val = val

def final_answer(val):
    raise FinalAnswerException(val)

namespace["final_answer"] = final_answer

def make_proxy_tool(tool_name, host_urls):
    def tool_proxy(*args, **kwargs):
        payload = json.dumps({"args": args, "kwargs": kwargs}).encode("utf-8")
        last_err = None
        for host_url in host_urls:
            try:
                req = urllib.request.Request(
                    f"{host_url}/tools/{tool_name}",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    return data.get("result")
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to invoke host tool '{tool_name}' via candidate URLs {host_urls}: {last_err}")
    return tool_proxy

def register_tools(tools_list, host_urls):
    for tool_name in tools_list:
        namespace[tool_name] = make_proxy_tool(tool_name, host_urls)

def run_code(code_str, max_output_len=50000):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured = io.StringIO()
    sys.stdout = captured
    sys.stderr = captured
    output = None
    is_final = False
    error = None
    try:
        parsed = ast.parse(code_str)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = parsed.body.pop()
            if parsed.body:
                exec(compile(parsed, "<codeact>", "exec"), namespace)
            output = eval(compile(ast.Expression(last_expr.value), "<codeact>", "eval"), namespace)
        else:
            exec(compile(parsed, "<codeact>", "exec"), namespace)
    except FinalAnswerException as fa:
        is_final = True
        output = fa.val
    except Exception as e:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    logs = captured.getvalue()
    if len(logs) > max_output_len:
        logs = logs[:max_output_len] + f"\\n... [truncated at {max_output_len} chars]"

    return {
        "output": repr(output) if output is not None else None,
        "raw_output": output if isinstance(output, (int, float, str, bool, list, dict)) else repr(output),
        "logs": logs,
        "is_final_answer": is_final,
        "error": error
    }

class WorkerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            action = body.get("action", "execute")

            if action == "register_tools":
                register_tools(body.get("tools", []), body.get("host_urls", []))
                res = {"status": "ok"}
            elif action == "set_variables":
                for k, v in body.get("variables", {}).items():
                    namespace[k] = v
                res = {"status": "ok"}
            elif action == "reset":
                saved_tools = {k: v for k, v in namespace.items() if callable(v) and k != "final_answer"}
                namespace.clear()
                namespace["__name__"] = "__main__"
                namespace["final_answer"] = final_answer
                namespace.update(saved_tools)
                res = {"status": "ok"}
            else:
                res = run_code(body.get("code", ""), body.get("max_output_len", 50000))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9199), WorkerHandler)
    server.serve_forever()
"""


class HostToolBridge:
    """HTTP bridge that allows in-container Python code to execute host tools."""

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self.tools = tools
        self.port = self._find_free_port()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start(self) -> None:
        if self._server is not None:
            return

        tools_ref = self.tools

        class BridgeHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    if not self.path.startswith("/tools/"):
                        self.send_response(404)
                        self.end_headers()
                        return

                    tool_name = self.path[len("/tools/") :]
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    args = payload.get("args", [])
                    kwargs = payload.get("kwargs", {})

                    if tool_name not in tools_ref:
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Tool '{tool_name}' not found"}).encode("utf-8"))
                        return

                    tool_fn = tools_ref[tool_name]
                    if isinstance(tool_fn, BaseTool):
                        if kwargs:
                            result = tool_fn.invoke(kwargs)
                        elif args and len(args) == 1 and isinstance(args[0], (str, dict)):
                            result = tool_fn.invoke(args[0])
                        else:
                            result = tool_fn.invoke(args[0] if args else "")
                    elif callable(tool_fn):
                        result = tool_fn(*args, **kwargs)
                    else:
                        result = str(tool_fn)

                    # Serialize result safely
                    if hasattr(result, "content"):
                        serialized = result.content
                    elif isinstance(result, (str, int, float, bool, list, dict)) or result is None:
                        serialized = result
                    else:
                        serialized = str(result)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"result": serialized}).encode("utf-8"))
                except Exception as exc:
                    logger.error(f"HostToolBridge error executing tool: {exc}")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        self._server = HTTPServer(("0.0.0.0", self.port), BridgeHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.debug(f"HostToolBridge listening on port {self.port}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
            logger.debug("HostToolBridge stopped")


class DockerPythonExecutor:
    """Stateful Python code executor running inside the all-in-one Docker sandbox."""

    def __init__(
        self,
        backend: AioSandboxBackend | None = None,
        additional_authorized_imports: list[str] | None = None,
        max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
        additional_functions: dict[str, Callable[..., Any]] | None = None,
        tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool] | None = None,
        timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
        initial_state: dict[str, Any] | None = None,
        fallback_to_local: bool = True,
    ) -> None:
        self.backend = backend
        self.additional_authorized_imports = additional_authorized_imports or []
        self.max_print_outputs_length = max_print_outputs_length
        self.timeout_seconds = timeout_seconds or MAX_EXECUTION_TIME_SECONDS
        self.initial_state = initial_state or {}
        self.additional_functions = additional_functions or {}
        self.fallback_to_local = fallback_to_local
        self.static_tools: dict[str, Callable[..., Any]] = {}
        self._host_bridge: HostToolBridge | None = None
        self._worker_ready = False
        self._local_fallback: Any | None = None

        self._init_tools(tools)

    def _get_local_fallback(self) -> Any:
        if self._local_fallback is None:
            from genai_tk.agents.tools.python_executor.executor import LocalPythonExecutor

            self._local_fallback = LocalPythonExecutor(
                additional_authorized_imports=self.additional_authorized_imports,
                tools=self.static_tools,
                max_print_outputs_length=self.max_print_outputs_length,
                timeout_seconds=self.timeout_seconds,
                initial_state=self.initial_state,
            )
        return self._local_fallback

    def _init_tools(self, tools: dict[str, Any] | list[Any] | None) -> None:
        adapted_tools: dict[str, Callable[..., Any]] = {}
        if isinstance(tools, list):
            resolved_tools: list[Any] = []
            for item in tools:
                if isinstance(item, BaseTool):
                    resolved_tools.append(item)
                elif isinstance(item, (str, dict)) or hasattr(item, "target") or hasattr(item, "tool_class"):
                    from genai_tk.agents.tools.langchain.shared_config_loader import process_langchain_tools_from_config

                    resolved = process_langchain_tools_from_config([item])
                    resolved_tools.extend(resolved)
                elif callable(item):
                    try:
                        res = item()
                        if isinstance(res, BaseTool):
                            resolved_tools.append(res)
                        elif isinstance(res, list) and all(isinstance(x, BaseTool) for x in res):
                            resolved_tools.extend(res)
                        else:
                            resolved_tools.append(item)
                    except Exception:
                        resolved_tools.append(item)
                else:
                    resolved_tools.append(item)

            for t in resolved_tools:
                if isinstance(t, BaseTool):
                    adapted_tools[t.name] = t
                elif callable(t):
                    name = getattr(t, "__name__", str(t))
                    adapted_tools[name] = t
        elif isinstance(tools, dict):
            for name, item in tools.items():
                if isinstance(item, BaseTool):
                    adapted_tools[name] = item
                elif isinstance(item, (str, dict)) or hasattr(item, "target") or hasattr(item, "tool_class"):
                    from genai_tk.agents.tools.langchain.shared_config_loader import process_langchain_tools_from_config

                    resolved = process_langchain_tools_from_config([item])
                    for t in resolved:
                        if isinstance(t, BaseTool):
                            adapted_tools[name] = t
                elif callable(item):
                    try:
                        res = item()
                        if isinstance(res, BaseTool):
                            adapted_tools[name] = res
                        else:
                            adapted_tools[name] = item
                    except Exception:
                        adapted_tools[name] = item

        self.static_tools = {**adapted_tools, **self.additional_functions}

    def send_tools(self, tools: dict[str, Any] | list[Any]) -> None:
        """Register or update tools available to the executor."""
        self._init_tools(tools)
        self._worker_ready = False  # re-sync tools on next call

    def send_variables(self, variables: dict[str, Any]) -> None:
        """Inject variables into the executor's state."""
        self.initial_state.update(variables)
        self._worker_ready = False

    async def _get_backend(self) -> AioSandboxBackend:
        if self.backend is not None:
            if not getattr(self.backend, "_sandbox", None):
                await self.backend.start()
            return self.backend
        return await DockerSandboxManager.aget_shared_backend()

    async def _write_file(self, backend: AioSandboxBackend, file_path: str, content: str) -> None:
        """Write or overwrite a file in the sandbox container."""
        if getattr(backend, "_sandbox", None) is not None:
            await backend._sandbox.files.write_file(file_path, content)
        else:
            await backend._run_write_file({"path": file_path, "content": content})

    async def _ensure_worker(self, backend: AioSandboxBackend) -> None:
        """Ensure the in-container python worker is running and configured."""
        # 1. Probe worker health
        health = await backend.aexecute("curl -s -m 1 http://127.0.0.1:9199/health")
        if health.output.strip() != "OK":
            # Upload worker script and launch daemon
            await self._write_file(backend, WORKER_SCRIPT_PATH, IN_CONTAINER_WORKER_CODE)
            await backend.aexecute(f"nohup python3 {WORKER_SCRIPT_PATH} >/tmp/pyworker.log 2>&1 &")
            # Poll health until ready
            for _ in range(20):
                await asyncio.sleep(0.1)
                h = await backend.aexecute("curl -s -m 1 http://127.0.0.1:9199/health")
                if h.output.strip() == "OK":
                    break

        # 2. Setup HostToolBridge if tools exist
        if self.static_tools:
            if self._host_bridge is None:
                self._host_bridge = HostToolBridge(self.static_tools)
                self._host_bridge.start()
            else:
                self._host_bridge.tools = self.static_tools

            bridge_port = self._host_bridge.port
            # Candidate host bridge URLs reachable from container
            candidate_urls = [
                f"http://172.17.0.1:{bridge_port}",
                f"http://172.18.0.1:{bridge_port}",
                f"http://172.25.240.1:{bridge_port}",
                f"http://172.25.253.111:{bridge_port}",
            ]
            reg_payload = json.dumps({
                "action": "register_tools",
                "tools": list(self.static_tools.keys()),
                "host_urls": candidate_urls,
            })
            await self._write_file(backend, "/tmp/reg_payload.json", reg_payload)
            await backend.aexecute(
                "curl -s -X POST http://127.0.0.1:9199/ -H 'Content-Type: application/json' --data-binary @/tmp/reg_payload.json"
            )

        # 3. Inject initial variables if any
        if self.initial_state:
            vars_payload = json.dumps({
                "action": "set_variables",
                "variables": {k: v for k, v in self.initial_state.items() if isinstance(v, (str, int, float, bool, list, dict))},
            })
            await self._write_file(backend, "/tmp/vars_payload.json", vars_payload)
            await backend.aexecute(
                "curl -s -X POST http://127.0.0.1:9199/ -H 'Content-Type: application/json' --data-binary @/tmp/vars_payload.json"
            )

        self._worker_ready = True

    async def aexecute_code(self, code: str) -> CodeOutput:
        """Asynchronously execute Python code in the Docker sandbox container."""
        try:
            backend = await self._get_backend()
            if not self._worker_ready:
                await self._ensure_worker(backend)

            exec_payload = json.dumps({
                "action": "execute",
                "code": code,
                "max_output_len": self.max_print_outputs_length,
            })
            await self._write_file(backend, "/tmp/exec_input.json", exec_payload)

            res = await backend.aexecute(
                "curl -s -X POST http://127.0.0.1:9199/ -H 'Content-Type: application/json' --data-binary @/tmp/exec_input.json"
            )

            if not res.output.strip():
                return CodeOutput(output=None, logs="", is_final_answer=False, error="Worker returned empty output")

            try:
                data = json.loads(res.output)
            except json.JSONDecodeError:
                raw = res.output.strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(raw[start : end + 1])
                else:
                    return CodeOutput(output=None, logs=raw, is_final_answer=False, error=f"Invalid worker JSON response: {raw}")

            raw_out = data.get("raw_output")
            is_final = data.get("is_final_answer", False)
            logs = data.get("logs", "")
            error = data.get("error")

            return CodeOutput(
                output=raw_out,
                logs=logs,
                is_final_answer=is_final,
                error=error,
            )
        except Exception as exc:
            if self.fallback_to_local:
                logger.warning(
                    f"Docker sandbox Python execution failed ({exc}); falling back to local Python executor."
                )
                try:
                    return self._get_local_fallback().execute_code(code)
                except Exception as local_exc:
                    logger.error(f"Local Python executor fallback exception: {local_exc}")
                    return CodeOutput(output=None, logs="", is_final_answer=False, error=str(local_exc))

            logger.error(f"DockerPythonExecutor exception: {exc}")
            return CodeOutput(output=None, logs="", is_final_answer=False, error=str(exc))

    def reset(self) -> None:
        """Reset the internal worker namespace."""
        if self._local_fallback is not None:
            self._local_fallback.reset()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            asyncio.run(self._areset())
        else:
            asyncio.run(self._areset())

    async def _areset(self) -> None:
        try:
            backend = await self._get_backend()
            await backend.aexecute(
                'curl -s -X POST http://127.0.0.1:9199/ -H "Content-Type: application/json" -d \'{"action": "reset"}\''
            )
        except Exception as exc:
            logger.debug(f"Error resetting worker: {exc}")

    def __call__(self, code: str) -> CodeOutput:
        """Synchronously execute Python code."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(self.aexecute_code(code))
        return asyncio.run(self.aexecute_code(code))

    def close(self) -> None:
        """Clean up host tool bridge."""
        if self._host_bridge is not None:
            self._host_bridge.stop()
            self._host_bridge = None
