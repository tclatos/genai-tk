"""Tests for SQL tool factory trace safety."""

from __future__ import annotations

import warnings
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.agents.tools.langchain.sql_tool_factory import create_sql_toolkit_tools


def test_sql_toolkit_tools_serialize_without_pydantic_warnings(fake_llm: BaseChatModel) -> None:
    """SQL toolkit tools can be captured by LangSmith without model warnings."""
    database_uri = f"sqlite:///{Path(__file__).resolve().parents[3] / 'data' / 'Chinook.sqlite'}"
    tools = create_sql_toolkit_tools(database_uri=database_uri, llm=fake_llm)
    checker = next(sql_tool for sql_tool in tools if sql_tool.name == "sql_db_query_checker")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        checker.model_dump(mode="json", fallback=str)

    assert not [warning for warning in caught if "Pydantic serializer warnings" in str(warning.message)]
