"""Data models for Python code execution."""

from typing import Any

from pydantic import BaseModel, Field


class CodeOutput(BaseModel):
    """Result of Python code execution."""

    output: Any = None
    logs: str = ""
    is_final_answer: bool = False
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class PythonExecutorInput(BaseModel):
    """Input schema for Python code execution tool."""

    code: str = Field(description="The Python code to execute in the in-process interpreter.")
