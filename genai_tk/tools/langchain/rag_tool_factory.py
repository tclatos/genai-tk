"""RAG Tool Factory for LangChain agents.

Creates LangChain tools that perform async similarity searches via a
:class:`~genai_tk.core.retriever_factory.ManagedRetriever`.

Configuration example (agent YAML profile)::

    tools:
      - spec: rag_search
        config:
          retriever: hybrid_ensemble
          tool_name: knowledge_search
          tool_description: "Search the company knowledge base"
          default_filter: {source: docs}
          top_k: 5
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class RAGToolConfig(BaseModel):
    """Configuration for the RAG tool factory.

    Example:
        ```python
        config = RAGToolConfig(
            retriever="hybrid_ensemble",
            tool_name="knowledge_base",
            tool_description="Search company knowledge base",
            default_filter={"department": "engineering"},
            top_k=5,
        )
        ```
    """

    retriever: str = Field(description="Retriever config tag (key in ``retrievers:`` YAML section)")
    tool_name: str = Field(default="rag_search", description="Name of the generated LangChain tool")
    tool_description: str = Field(
        default=(
            "Search the knowledge base for relevant documents. "
            "Accepts a query string and an optional metadata filter in JSON format."
        ),
        description="Description shown to the LLM agent",
    )
    default_filter: dict[str, Any] | None = Field(
        default=None,
        description="Default metadata filter merged with any runtime filter",
    )
    top_k: int = Field(default=4, description="Maximum number of results to return")


class RAGToolFactory:
    """Factory for creating RAG search tools backed by a ManagedRetriever.

    Example:
        ```python
        config = RAGToolConfig(
            retriever="hybrid_ensemble",
            tool_name="search_docs",
            tool_description="Search technical documentation",
            top_k=5,
        )
        factory = RAGToolFactory(llm)
        search_tool = factory.create_tool(config)
        result = await search_tool.ainvoke({"query": "vector search", "filter": '{"source": "docs"}'})
        ```
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def create_tool(self, config: RAGToolConfig) -> BaseTool:
        """Create a RAG tool from the given configuration.

        Args:
            config: Tool configuration including retriever tag and metadata.

        Returns:
            Async LangChain tool that accepts ``query`` and optional ``filter``.
        """
        from genai_tk.core.retriever_factory import RetrieverFactory

        try:
            managed = RetrieverFactory.create(config.retriever)
        except ValueError as exc:
            raise ValueError(f"Failed to create retriever '{config.retriever}': {exc}") from exc

        default_filter = config.default_filter
        top_k = config.top_k

        @tool
        async def rag_search_tool(query: str, filter: str | None = None) -> str:  # noqa: A002
            """Search the knowledge base for relevant documents.

            Args:
                query: The search query string.
                filter: Optional metadata filter as JSON (e.g. ``'{"source": "docs"}'``).

            Returns:
                Formatted string with matching document content and metadata.
            """
            # Parse runtime filter
            runtime_filter: dict[str, Any] | None = None
            if filter:
                try:
                    runtime_filter = json.loads(filter)
                except json.JSONDecodeError as exc:
                    return f"Error: invalid filter JSON — {exc}"

            # Merge filters
            merged: dict[str, Any] | None = None
            if default_filter or runtime_filter:
                merged = dict(default_filter or {})
                if runtime_filter:
                    merged.update(runtime_filter)

            try:
                docs = await managed.aquery(query, k=top_k, filter=merged)
            except Exception as exc:  # noqa: BLE001
                return f"Error searching knowledge base: {exc}"

            if not docs:
                return "No relevant documents found."

            parts: list[str] = []
            for i, doc in enumerate(docs, 1):
                parts.append(f"Document {i}:\n{doc.page_content}")
                if doc.metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in doc.metadata.items())
                    parts.append(f"Metadata: {meta_str}")
            return "\n\n".join(parts)

        rag_search_tool.name = config.tool_name
        rag_search_tool.description = config.tool_description
        return rag_search_tool

    def create_tool_from_dict(self, config_dict: dict[str, Any]) -> BaseTool:
        """Create a RAG tool from a plain dictionary.

        Args:
            config_dict: Dictionary matching :class:`RAGToolConfig` fields.

        Returns:
            Configured LangChain tool.
        """
        return self.create_tool(RAGToolConfig(**config_dict))


def create_rag_tool_from_config(config: dict[str, Any], llm: BaseChatModel | str = "default") -> BaseTool:
    """Convenience function to create a RAG tool from a config dict.

    Args:
        config: Tool configuration dict (see :class:`RAGToolConfig`).
        llm: LangChain chat model or identifier string (``"default"`` for the
            configured default LLM).

    Returns:
        Configured async RAG search tool.

    Example:
        ```python
        tool = create_rag_tool_from_config(
            {
                "retriever": "hybrid_ensemble",
                "tool_name": "knowledge_search",
                "top_k": 5,
            }
        )
        ```
    """
    if isinstance(llm, str):
        from genai_tk.core.llm_factory import get_llm

        llm = get_llm(llm if llm != "default" else None)

    return RAGToolFactory(llm).create_tool(RAGToolConfig(**config))
