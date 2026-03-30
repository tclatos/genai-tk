"""Shared fixtures and configuration for agent evaluation tests.

Provides:
- ``eval_agent`` fixture — a ``LangchainAgent`` with a fake LLM and a
  lightweight custom tool, ready for deterministic trajectory assertions.
- ``calculator_tool`` / ``echo_tool`` — simple tools with predictable outputs.
- ``judge_llm`` fixture — the configured ``fast_model`` ready for openevals.
  Skips automatically if ``--include-real-models`` is not set.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Minimal deterministic tools — no external calls, outputs are predictable
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the result.

    Args:
        expression: A Python arithmetic expression, e.g. '2 + 2'.
    """
    import ast
    import operator as op

    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def _eval(node: ast.AST) -> float:
        match node:
            case ast.Expression(body=body):
                return _eval(body)
            case ast.Constant(value=value) if isinstance(value, (int, float)):
                return float(value)
            case ast.BinOp(left=left, op=bop, right=right):
                return ops[type(bop)](_eval(left), _eval(right))
            case ast.UnaryOp(op=uop, operand=operand):
                return ops[type(uop)](_eval(operand))
            case _:
                raise ValueError(f"Unsupported node: {node}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def echo(message: str) -> str:
    """Echo back the message unchanged.

    Args:
        message: Any string to echo.
    """
    return message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator_tool():
    """Return the calculator tool for use in eval agent construction."""
    return calculator


@pytest.fixture
def echo_tool():
    """Return the echo tool for use in eval agent construction."""
    return echo


@pytest.fixture
def eval_agent(calculator_tool, echo_tool):
    """LangchainAgent with a fake LLM and deterministic tools.

    Uses ``parrot_local@fake`` — zero cost, no network, instant.
    """
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    return LangchainAgent(
        llm="parrot_local@fake",
        agent_type="react",
        tools=[calculator_tool, echo_tool],
    )


@pytest.fixture
def judge_llm(request):
    """Return the configured ``fast_model`` for use as an openevals judge.

    Skips automatically if ``--include-real-models`` is not set.
    """
    if not request.config.getoption("--include-real-models", default=False):
        pytest.skip("LLM-judged evals require --include-real-models")

    from genai_tk.core.llm_factory import get_llm

    return get_llm("fast_model")
