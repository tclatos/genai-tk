"""Unit tests for the in-process Python executor tool."""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.tools import tool

from genai_tk.agents.tools.python_executor import (
    LocalPythonExecutor,
    create_python_executor_tool,
)


@pytest.mark.unit
def test_basic_arithmetic_and_builtins() -> None:
    executor = LocalPythonExecutor()
    res = executor("import math\nx = 10 + 5 * 2\ny = math.sqrt(x)\nz = sqrt(16)\n(y, z)")
    assert res.error is None
    assert res.output[0] == pytest.approx(4.47213595499958)
    assert res.output[1] == 4.0


@pytest.mark.unit
def test_print_capture_and_formatting() -> None:
    executor = LocalPythonExecutor()
    res = executor("print('Hello', 'World')\nprint(123)\n42")
    assert res.error is None
    assert "Hello World\n123\n" == res.logs
    assert res.output == 42


@pytest.mark.unit
def test_state_persistence() -> None:
    executor = LocalPythonExecutor()
    res1 = executor("a = 100\nb = 200")
    assert res1.error is None
    res2 = executor("c = a + b\nc")
    assert res2.error is None
    assert res2.output == 300


@pytest.mark.unit
def test_functions_and_classes() -> None:
    executor = LocalPythonExecutor()
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Counter:
    def __init__(self, start=0):
        self.count = start

    def inc(self, step=1):
        self.count += step
        return self.count

c = Counter(10)
c.inc(5)
res = (factorial(5), c.count)
res
"""
    res = executor(code)
    assert res.error is None
    assert res.output == (120, 15)


@pytest.mark.unit
def test_forbidden_import_blocked() -> None:
    executor = LocalPythonExecutor()
    res = executor("import os\nos.getcwd()")
    assert res.error is not None
    assert "Import of 'os' is not allowed" in res.error


@pytest.mark.unit
def test_forbidden_dunder_attribute_blocked() -> None:
    executor = LocalPythonExecutor()
    res = executor("x = ().__class__.__bases__[0].__subclasses__()")
    assert res.error is not None
    assert "Forbidden access to dunder attribute" in res.error


@pytest.mark.unit
def test_langchain_custom_tool_integration() -> None:
    @tool
    def search_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: 22C, sunny"

    executor = LocalPythonExecutor(tools=[search_weather])
    code = """
msg = search_weather(city="Paris")
upper_msg = msg.upper()
upper_msg
"""
    res = executor(code)
    assert res.error is None
    assert res.output == "WEATHER IN PARIS: 22C, SUNNY"


@pytest.mark.unit
def test_pandas_dataframe_creation_and_manipulation() -> None:
    executor = LocalPythonExecutor(additional_authorized_imports=["pandas"])
    code = """
import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 35, 40],
    "salary": [50000, 65000, 70000, 85000],
    "department": ["Engineering", "HR", "Engineering", "Marketing"]
}

df = pd.DataFrame(data)
eng_df = df[df["department"] == "Engineering"]
avg_salary = float(eng_df["salary"].mean())
total_count = int(len(df))
(avg_salary, total_count)
"""
    res = executor(code)
    assert res.error is None
    assert res.output == (60000.0, 4)


@pytest.mark.unit
def test_pandas_preinjected_dataframe_and_operations() -> None:
    df_initial = pd.DataFrame(
        {
            "product": ["Widget A", "Widget B", "Widget C"],
            "price": [19.99, 29.99, 9.99],
            "quantity": [10, 5, 20],
        }
    )

    executor = LocalPythonExecutor(
        additional_authorized_imports=["pandas"],
        initial_state={"df": df_initial},
    )

    code = """
import pandas as pd

df["revenue"] = df["price"] * df["quantity"]
top_product = df.sort_values(by="revenue", ascending=False).iloc[0]["product"]
total_revenue = float(df["revenue"].sum())
(top_product, round(total_revenue, 2))
"""
    res = executor(code)
    assert res.error is None
    assert res.output == ("Widget A", 549.65)


@pytest.mark.unit
def test_python_executor_tool_sync() -> None:
    tool = create_python_executor_tool(additional_authorized_imports=["pandas"])
    code = """
import pandas as pd

series = pd.Series([1, 2, 3, 4, 5])
print("Series summary:")
print(series.describe())
int(series.sum())
"""
    result = tool.invoke({"code": code})
    assert "Series summary:" in result
    assert "Result:\n15" in result


@pytest.mark.asyncio
@pytest.mark.unit
async def test_python_executor_tool_async() -> None:
    tool = create_python_executor_tool(authorized_imports=["pandas"])
    code = """
import pandas as pd

df = pd.DataFrame({"x": [10, 20, 30], "y": [1, 2, 3]})
df["z"] = df["x"] + df["y"]
int(df["z"].max())
"""
    result = await tool.ainvoke({"code": code})
    assert "Result:\n33" in result


@pytest.mark.unit
def test_pandas_groupby_and_apply() -> None:
    executor = LocalPythonExecutor(additional_authorized_imports=["pandas"])
    code = """
import pandas as pd

df = pd.DataFrame({
    "group": ["A", "A", "B", "B", "B"],
    "val": [10, 20, 100, 200, 300]
})

grouped = df.groupby("group")["val"].mean()
dict(grouped)
"""
    res = executor(code)
    assert res.error is None
    assert res.output == {"A": 15.0, "B": 200.0}


@pytest.mark.unit
def test_error_handling_and_recovery() -> None:
    executor = LocalPythonExecutor()
    res = executor("x = 10 / 0")
    assert res.error is not None
    assert "ZeroDivisionError" in res.error

    # Executor still works afterwards
    res2 = executor("y = 42\ny")
    assert res2.error is None
    assert res2.output == 42


@pytest.mark.unit
def test_while_loop_limit() -> None:
    executor = LocalPythonExecutor()
    code = """
i = 0
while True:
    i += 1
"""
    res = executor(code)
    assert res.error is not None
    assert "iterations in while loop exceeded" in res.error


@pytest.mark.unit
def test_executor_reset() -> None:
    executor = LocalPythonExecutor()
    executor("secret_var = 'keep_it'")
    assert "secret_var" in executor.state
    executor.reset()
    assert "secret_var" not in executor.state
