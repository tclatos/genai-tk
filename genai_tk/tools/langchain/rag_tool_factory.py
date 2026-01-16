"""RAG Tool Factory for LangChain Integration.

This module provides a factory for creating RAG (Retrieval-Augmented Generation)
tools that can be used with LangChain agents. The factory creates tools that perform
similarity searches against vector stores with optional metadata filtering.
"""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from genai_tk.core.embeddings_store import EmbeddingsStore


class RAGToolConfig(BaseModel):
    """Configuration for RAG tool factory.

    Example:
        ```
        config = RAGToolConfig(
            embeddings_store="default",
            tool_name="knowledge_base",
            tool_description="Search company knowledge base for relevant documents",
            default_filter={"source": "docs"},
            top_k=5
        )
        ```
    """

    embeddings_store: str = Field(description="Name of vector store configuration to use")
    tool_name: str = Field(default="rag_search", description="Name of the generated tool")
    tool_description: str = Field(
        default="Search vector store for relevant documents. Accepts a query string and optional metadata filter in JSON format.",
        description="Description of what the tool does",
    )
    default_filter: dict[str, Any] | None = Field(
        default=None, description="Default metadata filter to apply to queries (merged with runtime filter)"
    )
    top_k: int = Field(default=4, description="Default maximum number of results to return")


class RAGToolFactory:
    """Factory for creating RAG (Retrieval-Augmented Generation) tools for LangChain agents.

    This factory creates tools that can execute similarity searches against vector stores
    using the EmbeddingsStore with optional metadata filtering capabilities.

    Example:
        ```
        config = RAGToolConfig(
            embeddings_store="knowledge_base",
            tool_name="search_docs",
            tool_description="Search technical documentation",
            default_filter={"category": "technical"},
            top_k=5
        )

        factory = RAGToolFactory(llm)
        tool = factory.create_tool(config)

        # The tool accepts a query and optional filter
        result = tool.invoke({"query": "Python best practices", "filter": '{"author": "John"}'})
        ```
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the factory with a language model.

        Args:
            llm: Language model (kept for compatibility, not currently used)
        """
        self.llm = llm

    def create_tool(self, config: RAGToolConfig) -> BaseTool:
        """Create a RAG tool based on the provided configuration.

        Args:
            config: Configuration specifying vector store and tool behavior

        Returns:
            Configured LangChain tool for RAG search that accepts query and optional filter

        Raises:
            ValueError: If embeddings store configuration is invalid
        """
        # Create embeddings store from configuration
        try:
            embeddings_store = EmbeddingsStore.create_from_config(config.embeddings_store)
        except Exception as e:
            raise ValueError(f"Failed to create embeddings store '{config.embeddings_store}': {e}") from e

        @tool
        async def rag_search_tool(query: str, filter: str | None = None) -> str:
            """Search vector store for relevant documents.

            Args:
                query: The search query string
                filter: Optional metadata filter as JSON string (e.g., '{"file_hash": "abc123"}')

            Returns:
                Formatted string containing relevant documents
            """
            try:
                # Parse runtime filter if provided
                runtime_filter = None
                if filter:
                    try:
                        runtime_filter = json.loads(filter)
                    except json.JSONDecodeError as e:
                        return f"Error: Invalid filter JSON format: {e}"

                # Merge default filter with runtime filter
                merged_filter = None
                if config.default_filter or runtime_filter:
                    merged_filter = {}
                    if config.default_filter:
                        merged_filter.update(config.default_filter)
                    if runtime_filter:
                        merged_filter.update(runtime_filter)

                # Perform similarity search using embeddings_store.query
                docs = await embeddings_store.query(query, k=config.top_k, filter=merged_filter)

                # Format documents into a single string
                if not docs:
                    return "No relevant documents found."

                result_parts = []
                for i, doc in enumerate(docs, 1):
                    result_parts.append(f"Document {i}:\n{doc.page_content}")
                    if doc.metadata:
                        metadata_str = ", ".join(f"{k}={v}" for k, v in doc.metadata.items())
                        result_parts.append(f"Metadata: {metadata_str}")

                return "\n\n".join(result_parts)
            except Exception as e:
                return f"Error searching vector store: {str(e)}"

        # Set the tool name and description after creation
        rag_search_tool.name = config.tool_name
        rag_search_tool.description = config.tool_description

        return rag_search_tool

    def create_tool_from_dict(self, config_dict: dict[str, Any]) -> BaseTool:
        """Create a RAG tool from a dictionary configuration.

        Args:
            config_dict: Dictionary containing tool configuration

        Returns:
            Configured LangChain tool for RAG search
        """
        config = RAGToolConfig(**config_dict)
        return self.create_tool(config)


def create_rag_tool_from_config(config: dict[str, Any], llm: BaseChatModel | None = None) -> BaseTool:
    """Create a RAG tool from a configuration dictionary.

    This function provides a simple interface for creating RAG tools
    from configuration files or dictionaries.

    Args:
        config: Configuration dictionary with tool settings
        llm: Language model (optional, will use default if not provided)

    Returns:
        Configured RAG search tool

    Example:
        ```
        config = {
            "embeddings_store": "knowledge_base",
            "tool_name": "search_documents",
            "tool_description": "Search company documents for relevant information",
            "default_filter": {"department": "engineering"},
            "top_k": 5
        }

        tool = create_rag_tool_from_config(config)

        # Use the tool with a query
        result = tool.invoke({"query": "What are the coding standards?"})

        # Use the tool with a query and additional filter
        result = tool.invoke({
            "query": "Python best practices",
            "filter": '{"author": "John Smith"}'
        })
        ```
    """
    # Get LLM if not provided
    if llm is None:
        from genai_tk.core.llm_factory import get_llm

        llm = get_llm()

    factory = RAGToolFactory(llm)
    return factory.create_tool_from_dict(config)
