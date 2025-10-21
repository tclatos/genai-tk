"""RAG Tool Factory for LangChain Integration.

This module provides a factory for creating RAG (Retrieval-Augmented Generation)
tools that can be used with LangChain agents. The factory creates tools that combine
vector store similarity search capabilities with configurable text splitting.
"""

import importlib
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
            tool_description="Search company knowledge base",
            text_splitter_config={
                "class": "RecursiveCharacterTextSplitter",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            filter_expression={"source": "docs"}
        )
        ```
    """

    embeddings_store: str = Field(description="Name of vector store registry config")
    tool_name: str = Field(default="rag_search", description="Name of the generated tool")
    tool_description: str = Field(
        default="Search vector store for relevant documents",
        description="Description of what the tool does",
    )
    text_splitter_config: dict[str, Any] = Field(
        default_factory=lambda: {"class": "RecursiveCharacterTextSplitter", "chunk_size": 1000},
        description="Text splitter configuration with class name and parameters",
    )
    filter_expression: dict[str, Any] | None = Field(
        default=None, description="Optional filter expression for vector store queries"
    )
    top_k: int = Field(default=5, description="Maximum number of results to return")

    def model_post_init(self, __context) -> None:
        """Validate text splitter configuration."""
        if "class" not in self.text_splitter_config:
            raise ValueError("text_splitter_config must contain a 'class' key")


class RAGToolFactory:
    """Factory for creating RAG (Retrieval-Augmented Generation) tools for LangChain agents.

    This factory creates tools that can execute similarity searches against vector stores
    using configurable text splitting and filtering capabilities.

    Example:
        ```
        config = RAGToolConfig(
            embeddings_store="knowledge_base",
            tool_name="search_docs",
            tool_description="Search technical documentation",
            text_splitter_config={
                "class": "RecursiveCharacterTextSplitter",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            filter_expression={"category": "technical"}
        )

        factory = RAGToolFactory(llm)
        tool = factory.create_tool(config)
        ```
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the factory with a language model.

        Args:
            llm: Language model for potential future use in result processing
        """
        self.llm = llm

    def create_tool(self, config: RAGToolConfig) -> BaseTool:
        """Create a RAG tool based on the provided configuration.

        Args:
            config: Configuration specifying vector store, text splitter, and tool behavior

        Returns:
            Configured LangChain tool for RAG search
        """
        # Create vector store registry
        try:
            embeddings_store = EmbeddingsStore.create_from_config(config.embeddings_store)
            vector_store = embeddings_store.get()
        except Exception as e:
            raise ValueError(f"Failed to create vector store from registry '{config.embeddings_store}': {e}") from e

        # Validate text splitter configuration
        try:
            splitter_class_name = config.text_splitter_config["class"]
            splitter_params = {k: v for k, v in config.text_splitter_config.items() if k != "class"}

            # Import text splitter class dynamically to validate it exists
            module = importlib.import_module("langchain.text_splitter")
            splitter_class = getattr(module, splitter_class_name)
            # Text splitter can be instantiated for future use if needed
            splitter_class(**splitter_params)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import text splitter class '{splitter_class_name}': {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to create text splitter: {e}") from e

        @tool
        async def rag_search_tool(query: str) -> str:
            """Search vector store for relevant documents."""
            try:
                # Perform similarity search
                kwargs = {"k": config.top_k}
                if config.filter_expression:
                    kwargs["filter"] = config.filter_expression

                docs = await vector_store.asimilarity_search(query, **kwargs)

                # Format documents into a single string
                if not docs:
                    return "No relevant documents found."

                result_parts = []
                for i, doc in enumerate(docs, 1):
                    result_parts.append(f"Document {i}: {doc.page_content}")

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
        llm: Language model for potential future use (optional, will use default if not provided)

    Returns:
        Configured RAG search tool

    Example:
        ```
        config = {
            "embeddings_store": "knowledge_base",
            "tool_name": "search_documents",
            "tool_description": "Search company documents",
            "text_splitter_config": {
                "class": "RecursiveCharacterTextSplitter",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            "filter_expression": {"department": "engineering"},
            "top_k": 3
        }

        tool = create_rag_tool_from_config(config)
        ```
    """
    # Get LLM if not provided
    if llm is None:
        from genai_tk.core.llm_factory import get_llm

        llm = get_llm()

    factory = RAGToolFactory(llm)
    return factory.create_tool_from_dict(config)
