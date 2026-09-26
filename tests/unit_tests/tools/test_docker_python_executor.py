"""Unit tests for DockerPythonExecutor and HostToolBridge (mocked and real)."""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.tools import tool

from genai_tk.agents.sandbox.manager import DockerSandboxManager
from genai_tk.agents.tools.python_executor.docker_executor import (
    DockerPythonExecutor,
    HostToolBridge,
)
from genai_tk.agents.tools.python_executor.tool import (
    create_python_executor_tool,
)


@pytest.mark.unit
def test_host_tool_bridge_lifecycle() -> None:
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    bridge = HostToolBridge({"add_numbers": add_numbers})
    bridge.start()
    assert bridge.port > 0
    bridge.stop()


@pytest.mark.unit
def test_docker_sandbox_manager_availability() -> None:
    # is_docker_available should return a boolean without raising
    avail = DockerSandboxManager.is_docker_available()
    assert isinstance(avail, bool)


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_arithmetic_and_numpy() -> None:
    executor = DockerPythonExecutor()
    code = """
import numpy as np
import math

x = 100
y = np.sqrt(x) * 2
int(y)
"""
    res = await executor.aexecute_code(code)
    assert res.error is None
    assert res.output in (20, "20", np.int64(20))


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_state_persistence() -> None:
    executor = DockerPythonExecutor()
    res1 = await executor.aexecute_code("val = 42\nprint('stored val')\nval")
    assert res1.error is None
    assert "stored val" in res1.logs

    res2 = await executor.aexecute_code("val_doubled = val * 2\nval_doubled")
    assert res2.error is None
    assert res2.output in (84, "84")


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_final_answer() -> None:
    executor = DockerPythonExecutor()
    code = """
speed = 58.0 / 3.6
length = 155.0
time = length / speed
final_answer({"time": round(time, 2), "unit": "seconds"})
"""
    res = await executor.aexecute_code(code)
    assert res.error is None
    assert res.is_final_answer is True
    assert "9.62" in str(res.output)


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_python_executor_host_tool_bridging() -> None:
    @tool
    def web_search(query: str) -> str:
        """Mock web search tool."""
        return f"Mock search info for {query}: length is 155 meters, speed is 58 km/h"

    executor = DockerPythonExecutor(tools=[web_search])
    code = """
info = web_search("Pont des Arts Paris")
print("Search info:", info)
res = "length is 155" in info
res
"""
    res = await executor.aexecute_code(code)
    assert res.error is None
    assert res.output is True
    executor.close()


@pytest.mark.docker
@pytest.mark.integration
def test_docker_python_executor_tool_sync_invoke() -> None:
    tool = create_python_executor_tool(executor_type="docker")
    code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
df["c"] = df["a"] + df["b"]
int(df["c"].max())
"""
    out = tool.invoke({"code": code})
    assert "Result:\n33" in out
