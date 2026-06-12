"""LangChain-based agent implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

__all__ = ["LangchainAgent"]


def __getattr__(name: str) -> object:
    if name == "LangchainAgent":
        from genai_tk.agents.langchain.langchain_agent import LangchainAgent

        return LangchainAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
