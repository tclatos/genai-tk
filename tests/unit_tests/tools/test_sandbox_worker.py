"""Exercise the standalone supervisor using harmless local subprocesses."""

from __future__ import annotations

import http.client
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from genai_tk.agents.tools.python_executor import sandbox_worker

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.name != "posix", reason="POSIX sandbox supervisor")]


class WorkerClient:
    def __init__(self, directory: Path, token: str, process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
        self.directory = directory
        self.token = token
        self.process = process
        self.timeout_seconds = timeout_seconds
        self.port = int((directory / "port").read_text())

    def request(self, payload: dict[str, object], *, token: str | None = None) -> dict[str, object]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token if token is None else token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds + 3) as response:
            return json.load(response)


@pytest.fixture
def worker(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[WorkerClient]:
    token = secrets.token_urlsafe(32)
    timeout_seconds = float(getattr(request, "param", 30))
    settings = {
        "token": token,
        "directory": str(tmp_path),
        "timeout_seconds": timeout_seconds,
        "max_output_chars": 128,
        "memory_limit_mb": 1024,
    }
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(settings))
    process = subprocess.Popen(
        [sys.executable, "-I", "-u", str(Path(sandbox_worker.__file__).resolve()), "--config", str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    for _ in range(100):
        if (tmp_path / "port").exists():
            break
        if process.poll() is not None:
            pytest.fail(process.stderr.read().decode())
        time.sleep(0.02)
    else:
        process.kill()
        pytest.fail("Worker did not become ready.")
    client = WorkerClient(tmp_path, token, process, timeout_seconds)
    try:
        yield client
    finally:
        try:
            client.request({"action": "shutdown"})
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def test_wrong_token_is_rejected(worker: WorkerClient) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        worker.request({"action": "execute", "code": "40 + 2"}, token="wrong-session")
    assert exc.value.code == 401


def test_state_persists_within_session(worker: WorkerClient) -> None:
    worker.request({"action": "execute", "code": "value = 40"})
    assert worker.request({"action": "execute", "code": "value + 2"})["output"] == 42


def test_native_numpy_and_pandas(worker: WorkerClient) -> None:
    result = worker.request(
        {"action": "execute", "code": "import numpy as np\nimport pandas as pd\nint(pd.Series(np.arange(4)).sum())"}
    )
    assert result.get("output") == 6, result


def test_final_answer_preserved(worker: WorkerClient) -> None:
    assert worker.request({"action": "execute", "code": "final_answer(42)"})["is_final_answer"] is True


def test_reset_clears_agent_defined_functions(worker: WorkerClient) -> None:
    worker.request({"action": "execute", "code": "def saved_secret():\n    return 42"})
    worker.request({"action": "reset"})
    assert worker.request({"action": "execute", "code": "'saved_secret' in globals()"})["output"] is False


def test_output_is_bounded_during_capture(worker: WorkerClient) -> None:
    result = worker.request({"action": "execute", "code": "for _ in range(10000):\n    print('abcdefghij')"})
    assert len(result["logs"]) == 128


@pytest.mark.parametrize("worker", [0.3], indirect=True)
def test_timeout_stops_late_side_effects(worker: WorkerClient, tmp_path: Path) -> None:
    marker = tmp_path / "late"
    result = worker.request(
        {
            "action": "execute",
            "code": f"import time\nfrom pathlib import Path\ntime.sleep(1)\nPath({str(marker)!r}).touch()",
        }
    )
    time.sleep(1.1)
    assert result["fatal"] is True and not marker.exists()


def test_tool_call_requires_matching_response(worker: WorkerClient) -> None:
    worker.request({"action": "configure", "tools": ["lookup"], "variables": {}})
    worker.request({"action": "execute", "code": "lookup('query')"})
    result = worker.request({"action": "tool_result", "request_id": "another-session", "result": 42})
    assert result["fatal"] is True


def test_tool_relay_resumes_same_execution(worker: WorkerClient) -> None:
    worker.request({"action": "configure", "tools": ["lookup"], "variables": {}})
    call = worker.request({"action": "execute", "code": "lookup('query') + 2"})
    result = worker.request({"action": "tool_result", "request_id": call["request_id"], "result": 40})
    assert result["output"] == 42


def test_tool_configuration_removes_old_names(worker: WorkerClient) -> None:
    worker.request({"action": "configure", "tools": ["old"], "variables": {}})
    worker.request({"action": "configure", "tools": [], "variables": {}})
    assert worker.request({"action": "execute", "code": "'old' in globals()"})["output"] is False


def test_excessive_request_is_rejected(worker: WorkerClient) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", worker.port, timeout=2)
    try:
        connection.putrequest("POST", "/")
        connection.putheader("Authorization", f"Bearer {worker.token}")
        connection.putheader("Content-Length", str(sandbox_worker.MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        assert connection.getresponse().status == 413
    finally:
        connection.close()


def test_authentication_precedes_body_read(worker: WorkerClient) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", worker.port, timeout=2)
    try:
        connection.putrequest("POST", "/")
        connection.putheader("Content-Length", "1024")
        connection.endheaders()
        assert connection.getresponse().status == 401
    finally:
        connection.close()


@pytest.mark.parametrize("worker", [0.3], indirect=True)
def test_blocked_worker_input_obeys_deadline(worker: WorkerClient) -> None:
    worker.request(
        {
            "action": "execute",
            "code": (
                "import sys, time\n"
                "class SlowInput:\n"
                "    @property\n"
                "    def buffer(self): return self\n"
                "    def readline(self, *args): time.sleep(30)\n"
                "sys.stdin = SlowInput()\n"
                "42"
            ),
        }
    )
    result = worker.request({"action": "execute", "code": " " * 200000})
    assert result["fatal"] is True


@pytest.mark.parametrize("worker", [0.3], indirect=True)
def test_timeout_stops_descendants_in_worker_process_group(worker: WorkerClient, tmp_path: Path) -> None:
    marker = tmp_path / "late-child"
    child_code = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    result = worker.request(
        {
            "action": "execute",
            "code": (
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                "time.sleep(10)"
            ),
        }
    )
    time.sleep(1.1)
    assert result["fatal"] is True and not marker.exists()
