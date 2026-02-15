"""Unit tests for RAG Tool Factory."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.tools.langchain.rag_tool_factory import (
    RAGToolConfig,
    RAGToolFactory,
    create_rag_tool_from_config,
)


class TestRAGToolConfig:
    """Test RAGToolConfig validation."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = RAGToolConfig(
            embeddings_store="test_registry",
            tool_name="test_tool",
            tool_description="Test description",
            default_filter={"category": "test"},
            top_k=3,
        )
        assert config.embeddings_store == "test_registry"
        assert config.tool_name == "test_tool"
        assert config.tool_description == "Test description"
        assert config.default_filter == {"category": "test"}
        assert config.top_k == 3

    def test_default_values(self):
        """Test default configuration values."""
        config = RAGToolConfig(embeddings_store="test")
        assert config.tool_name == "rag_search"
        assert "Search vector store for relevant documents" in config.tool_description
        assert config.default_filter is None
        assert config.top_k == 4

    def test_none_default_filter(self):
        """Test that None default filter is allowed."""
        config = RAGToolConfig(embeddings_store="test", default_filter=None)
        assert config.default_filter is None


class TestRAGToolFactory:
    """Test RAGToolFactory functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock language model."""
        return Mock(spec=BaseChatModel)

    @pytest.fixture
    def factory(self, mock_llm):
        """Create a RAGToolFactory instance."""
        return RAGToolFactory(mock_llm)

    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration."""
        return RAGToolConfig(
            embeddings_store="test_registry",
            tool_name="test_search",
            tool_description="Test search tool",
            top_k=3,
        )

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    def test_create_tool_success(self, mock_registry_class, factory, basic_config):
        """Test successful tool creation."""
        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_embeddings_store.query = AsyncMock(
            return_value=[Document(page_content="First document"), Document(page_content="Second document")]
        )
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create tool
        tool = factory.create_tool(basic_config)

        # Verify tool properties
        assert tool.name == "test_search"
        assert tool.description == "Test search tool"

        # Verify embeddings store was created correctly
        mock_registry_class.create_from_config.assert_called_once_with("test_registry")

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    def test_create_tool_embeddings_store_error(self, mock_registry_class, factory, basic_config):
        """Test error when embeddings store creation fails."""
        mock_registry_class.create_from_config.side_effect = Exception("Embeddings store error")

        with pytest.raises(ValueError, match="Failed to create embeddings store 'test_registry'"):
            factory.create_tool(basic_config)

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_success(self, mock_registry_class, factory, basic_config):
        """Test tool ainvoke returns formatted documents."""
        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_docs = [
            Document(page_content="First document content"),
            Document(page_content="Second document content"),
        ]
        mock_embeddings_store.query = AsyncMock(return_value=mock_docs)
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        # Verify query was called correctly
        mock_embeddings_store.query.assert_called_once_with("test query", k=3, filter=None)

        # Verify result formatting
        assert "Document 1:" in result
        assert "First document content" in result
        assert "Document 2:" in result
        assert "Second document content" in result

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_with_filter(self, mock_registry_class, factory):
        """Test tool ainvoke with filter expression."""
        config = RAGToolConfig(embeddings_store="test_registry", default_filter={"category": "technical"}, top_k=2)

        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_docs = [Document(page_content="Filtered document")]
        mock_embeddings_store.query = AsyncMock(return_value=mock_docs)
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create and invoke tool
        tool = factory.create_tool(config)
        result = await tool.ainvoke({"query": "test query"})

        # Verify query was called with filter
        mock_embeddings_store.query.assert_called_once_with("test query", k=2, filter={"category": "technical"})

        assert "Document 1:" in result
        assert "Filtered document" in result

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_with_runtime_filter(self, mock_registry_class, factory, basic_config):
        """Test tool ainvoke with runtime filter passed as JSON string."""
        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_docs = [Document(page_content="Runtime filtered document")]
        mock_embeddings_store.query = AsyncMock(return_value=mock_docs)
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create and invoke tool with runtime filter
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query", "filter": '{"author": "John"}'})

        # Verify query was called with merged filter
        mock_embeddings_store.query.assert_called_once_with("test query", k=3, filter={"author": "John"})

        assert "Document 1:" in result

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_no_results(self, mock_registry_class, factory, basic_config):
        """Test tool ainvoke when no documents are found."""
        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_embeddings_store.query = AsyncMock(return_value=[])
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert result == "No relevant documents found."

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_search_error(self, mock_registry_class, factory, basic_config):
        """Test tool ainvoke handles search errors gracefully."""
        # Mock embeddings store
        mock_embeddings_store = Mock()
        mock_embeddings_store.query = AsyncMock(side_effect=Exception("Search failed"))
        mock_registry_class.create_from_config.return_value = mock_embeddings_store

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert "Error searching vector store:" in result
        assert "Search failed" in result

    def test_create_tool_from_dict(self, factory):
        """Test creating tool from dictionary configuration."""
        config_dict = {
            "embeddings_store": "test_registry",
            "tool_name": "dict_tool",
            "tool_description": "Tool from dict",
            "top_k": 4,
        }

        with patch.object(factory, "create_tool") as mock_create_tool:
            factory.create_tool_from_dict(config_dict)

            # Verify create_tool was called with correct config
            args, _ = mock_create_tool.call_args
            config = args[0]
            assert isinstance(config, RAGToolConfig)
            assert config.embeddings_store == "test_registry"
            assert config.tool_name == "dict_tool"
            assert config.top_k == 4


class TestCreateRAGToolFromConfig:
    """Test convenience function create_rag_tool_from_config."""

    @patch("genai_tk.tools.langchain.rag_tool_factory.RAGToolFactory")
    @patch("genai_tk.core.llm_factory.get_llm")
    def test_create_rag_tool_from_config_with_default_llm(self, mock_get_llm, mock_factory_class):
        """Test creating RAG tool with default LLM."""
        mock_llm = Mock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm

        mock_factory = Mock()
        mock_tool = Mock()
        mock_factory.create_tool_from_dict.return_value = mock_tool
        mock_factory_class.return_value = mock_factory

        config = {"embeddings_store": "test"}

        result = create_rag_tool_from_config(config)

        # Verify LLM was fetched and factory was created
        mock_get_llm.assert_called_once()
        mock_factory_class.assert_called_once_with(mock_llm)
        mock_factory.create_tool_from_dict.assert_called_once_with(config)
        assert result == mock_tool

    @patch("genai_tk.tools.langchain.rag_tool_factory.RAGToolFactory")
    def test_create_rag_tool_from_config_with_provided_llm(self, mock_factory_class):
        """Test creating RAG tool with provided LLM."""
        mock_llm = Mock(spec=BaseChatModel)

        mock_factory = Mock()
        mock_tool = Mock()
        mock_factory.create_tool_from_dict.return_value = mock_tool
        mock_factory_class.return_value = mock_factory

        config = {"embeddings_store": "test"}

        result = create_rag_tool_from_config(config, llm=mock_llm)

        # Verify factory was created with provided LLM
        mock_factory_class.assert_called_once_with(mock_llm)
        mock_factory.create_tool_from_dict.assert_called_once_with(config)
        assert result == mock_tool
