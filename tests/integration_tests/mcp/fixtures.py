"""Tool factories for MCP integration tests.

These are simple, dependency-free LangChain tools used by test YAML configs
so that integration tests do not require real API keys or external services.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Echo tool – single string input
# ---------------------------------------------------------------------------


class _EchoTool(BaseTool):
    name: str = "echo_text"
    description: str = "Echo back the input text."

    def _run(self, text: str) -> str:
        return text

    async def _arun(self, text: str) -> str:
        return text


def create_echo_tool() -> BaseTool:
    """Factory: returns an echo tool."""
    return _EchoTool()


# ---------------------------------------------------------------------------
# Reverse tool – single string input
# ---------------------------------------------------------------------------


class _ReverseTool(BaseTool):
    name: str = "reverse_text"
    description: str = "Return the reverse of the input string."

    def _run(self, text: str) -> str:
        return text[::-1]

    async def _arun(self, text: str) -> str:
        return text[::-1]


def create_reverse_tool() -> BaseTool:
    """Factory: returns a reverse-text tool."""
    return _ReverseTool()


# ---------------------------------------------------------------------------
# Add tool – structured (two-field Pydantic schema)
# ---------------------------------------------------------------------------


class _AddSchema(BaseModel):
    a: int
    b: int = 0


class _AddTool(BaseTool):
    name: str = "add_numbers"
    description: str = "Return the sum of a and b."
    args_schema: type[BaseModel] = _AddSchema

    def _run(self, a: int, b: int = 0) -> int:
        return a + b

    async def _arun(self, a: int, b: int = 0) -> int:
        return a + b


def create_add_tool() -> BaseTool:
    """Factory: returns an addition tool with two integer parameters."""
    return _AddTool()
