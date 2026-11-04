"""Standalone, standard-library-only supervisor uploaded to the sandbox.

The surrounding container provides isolation; this supervisor provides bounded
transport, session state and killable execution. Worker messages are JSON.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import secrets
import select
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO

MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4_194_304


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _decode(value: str | bytes) -> dict[str, object]:
    decoded = json.loads(value, parse_constant=_reject_constant)
    if not isinstance(decoded, dict):
        raise ValueError("A JSON object is required.")
    return decoded


def _encode(value: dict[str, object], limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Bound encoding incrementally instead of retaining arbitrarily large results."""
    chunks: list[bytes] = []
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(value):
        encoded = chunk.encode("utf-8")
        size += len(encoded)
        if size > limit:
            raise ValueError("Worker response exceeds the configured size limit.")
        chunks.append(encoded)
    return b"".join(chunks) + b"\n"


class BoundedOutput(io.TextIOBase):
    """Keep at most the configured number of output characters."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.length = 0

    def write(self, value: str) -> int:
        kept = value[: self.limit - self.length]
        if kept:
            self.parts.append(kept)
        self.length += len(kept)
        return len(value)

    def getvalue(self) -> str:
        return "".join(self.parts)

    def flush(self) -> None:
        pass


class FinalAnswer(BaseException):
    def __init__(self, value: object) -> None:
        self.value = value


def _final_answer(value: object) -> None:
    raise FinalAnswer(value)


def _child_main(settings: dict[str, object]) -> None:
    # Native stdout/stderr must not fill the supervisor's protocol pipe.
    # Python print is captured separately; native library writes are discarded.
    protocol: TextIO = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    if sys.platform == "linux":
        import resource

        memory = int(settings["memory_limit_mb"]) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))

    namespace: dict[str, object] = {"__name__": "__main__", "final_answer": _final_answer}
    registered: set[str] = set()

    def emit(message: dict[str, object]) -> None:
        protocol.write(_encode(message).decode("utf-8"))
        protocol.flush()

    def proxy(name: str):
        def call(*args: object, **kwargs: object) -> object:
            emit({"kind": "tool_call", "tool_name": name, "args": args, "kwargs": kwargs})
            raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Host tool response is too large.")
            response = _decode(raw)
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            return response.get("result")

        return call

    while raw := sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1):
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Worker command is too large.")
        request = _decode(raw)
        action = request.get("action")
        if action == "configure":
            for name in registered:
                namespace.pop(name, None)
            registered = set(request.get("tools", []))
            for name in registered:
                namespace[name] = proxy(name)
            namespace.update(request.get("variables", {}))
            emit({"kind": "ready"})
            continue
        if action != "execute":
            raise ValueError("Unexpected worker command.")

        output = BoundedOutput(int(settings["max_output_chars"]))
        value: object = None
        error = None
        final = False
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                parsed = ast.parse(str(request["code"]))
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    expression = parsed.body.pop()
                    exec(compile(parsed, "<codeact>", "exec"), namespace)
                    value = eval(compile(ast.Expression(expression.value), "<codeact>", "eval"), namespace)
                else:
                    exec(compile(parsed, "<codeact>", "exec"), namespace)
        except FinalAnswer as exc:
            final = True
            value = exc.value
        except BaseException:
            error = traceback.format_exc(limit=8)[-8192:]
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            value = repr(value)
        try:
            emit(
                {"kind": "result", "output": value, "logs": output.getvalue(), "error": error, "is_final_answer": final}
            )
        except (TypeError, ValueError, RecursionError):
            emit({"kind": "result", "error": "Result is not bounded JSON-compatible data."})


class WorkerSession:
    """Supervise one persistent child with a deadline independent of its code."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None
        self.selector = selectors.DefaultSelector()
        self.buffer = bytearray()
        self.deadline: float | None = None
        self.pending_id: str | None = None
        self.timed_out = False
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self) -> None:
        while not self.stopped.wait(0.05):
            # Active requests enforce their own deadline while holding this
            # lock. The watchdog covers waiting for a host tool response.
            if self.lock.acquire(blocking=False):
                try:
                    deadline = self.deadline
                    if deadline is not None and time.monotonic() >= deadline:
                        self.timed_out = True
                        self.stop_child()
                finally:
                    self.lock.release()

    def start_child(self) -> None:
        if self.process is not None:
            return
        self.timed_out = False
        self.buffer.clear()
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self._send(self.settings)

    def stop_child(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                with contextlib.suppress(KeyError, ValueError):
                    self.selector.unregister(stream)
                stream.close()
        self.deadline = None
        self.pending_id = None
        self.buffer.clear()

    def _send(self, request: dict[str, object]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("Worker process is unavailable.")
        data = memoryview(_encode(request, MAX_REQUEST_BYTES))
        descriptor = process.stdin.fileno()
        while data:
            remaining = (self.deadline or 0) - time.monotonic()
            if remaining <= 0:
                self.timed_out = True
                raise TimeoutError("Worker input exceeded its deadline.")
            if select.select([], [descriptor], [], min(remaining, 0.05))[1]:
                try:
                    written = os.write(descriptor, data[:65536])
                    data = data[written:]
                except BlockingIOError:
                    continue

    def _next(self) -> dict[str, object]:
        while True:
            deadline = self.deadline
            if deadline is None or time.monotonic() >= deadline:
                self.timed_out = True
                self.stop_child()
                return {"kind": "result", "fatal": True, "error": "Python execution exceeded its deadline."}
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                message = _decode(bytes(self.buffer[:newline]))
                del self.buffer[: newline + 1]
                if message.get("kind") == "tool_call":
                    self.pending_id = secrets.token_hex(16)
                    message["request_id"] = self.pending_id
                else:
                    self.deadline = None
                return message
            for key, _mask in self.selector.select(timeout=0.05):
                try:
                    chunk = os.read(key.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    raise RuntimeError("Worker process exited before responding.")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE_BYTES:
                    raise ValueError("Worker protocol response is too large.")

    def handle(self, request: dict[str, object]) -> dict[str, object]:
        with self.lock:
            action = request.get("action")
            if action in ("reset", "shutdown"):
                self.stop_child()
                self.timed_out = False
                return {"kind": "ready"}
            if self.timed_out:
                return {"kind": "result", "fatal": True, "error": "Python execution exceeded its deadline."}
            try:
                if action in ("execute", "configure"):
                    if self.pending_id is not None:
                        raise ValueError("A host tool result is required before another command.")
                    self.deadline = time.monotonic() + float(self.settings["timeout_seconds"])
                    self.start_child()
                    self._send(request)
                elif action == "tool_result":
                    supplied_id = request.get("request_id")
                    if self.pending_id is None or supplied_id != self.pending_id:
                        raise ValueError("Unexpected host tool response.")
                    self.pending_id = None
                    self._send(request)
                else:
                    raise ValueError("Unknown worker action.")
                return self._next()
            except (OSError, ValueError, RuntimeError) as exc:
                timed_out = self.timed_out
                self.stop_child()
                return {
                    "kind": "result",
                    "fatal": True,
                    "error": "Python execution exceeded its deadline." if timed_out else str(exc),
                }

    def close(self) -> None:
        self.stopped.set()
        with self.lock:
            self.stop_child()
            self.selector.close()


def serve(settings: dict[str, object]) -> None:
    session = WorkerSession(settings)
    token = str(settings["token"]).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def do_POST(self) -> None:
            auth = self.headers.get("Authorization", "").encode("utf-8")
            if not secrets.compare_digest(auth, b"Bearer " + token):
                self.send_error(401)
                self.close_connection = True
                return
            if self.path != "/":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= length <= MAX_REQUEST_BYTES:
                    self.send_error(413)
                    self.close_connection = True
                    return
                request = _decode(self.rfile.read(length))
                response = _encode(session.handle(request))
            except (ValueError, OSError, RecursionError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            if request.get("action") == "shutdown":
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    Path(str(settings["directory"]), "port").write_text(str(server.server_port))
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        session.close()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--child", action="store_true")
    options = parser.parse_args()
    if options.child:
        _child_main(_decode(sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)))
    else:
        settings = _decode(Path(options.config).read_text())
        timeout = float(settings["timeout_seconds"])
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("Invalid execution timeout.")
        serve(settings)


if __name__ == "__main__":
    main()
