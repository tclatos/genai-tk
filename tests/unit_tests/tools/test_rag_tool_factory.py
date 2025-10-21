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
            text_splitter_config={"class": "RecursiveCharacterTextSplitter", "chunk_size": 500, "chunk_overlap": 50},
            filter_expression={"category": "test"},
            top_k=3,
        )
        assert config.embeddings_store == "test_registry"
        assert config.tool_name == "test_tool"
        assert config.tool_description == "Test description"
        assert config.text_splitter_config["class"] == "RecursiveCharacterTextSplitter"
        assert config.filter_expression == {"category": "test"}
        assert config.top_k == 3

    def test_default_values(self):
        """Test default configuration values."""
        config = RAGToolConfig(embeddings_store="test")
        assert config.tool_name == "rag_search"
        assert config.tool_description == "Search vector store for relevant documents"
        assert config.text_splitter_config["class"] == "RecursiveCharacterTextSplitter"
        assert config.text_splitter_config["chunk_size"] == 1000
        assert config.filter_expression is None
        assert config.top_k == 5

    def test_missing_splitter_class(self):
        """Test validation error when text splitter class is missing."""
        with pytest.raises(ValueError, match="text_splitter_config must contain a 'class' key"):
            RAGToolConfig(embeddings_store="test", text_splitter_config={"chunk_size": 500})

    def test_none_filter_expression(self):
        """Test that None filter expression is allowed."""
        config = RAGToolConfig(embeddings_store="test", filter_expression=None)
        assert config.filter_expression is None


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
            text_splitter_config={"class": "RecursiveCharacterTextSplitter", "chunk_size": 500},
            top_k=3,
        )

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    def test_create_tool_success(self, mock_importlib, mock_registry_class, factory, basic_config):
        """Test successful tool creation."""
        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = AsyncMock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock text splitter
        mock_module = Mock()
        mock_splitter_class = Mock()
        mock_splitter = Mock()
        mock_module.RecursiveCharacterTextSplitter = mock_splitter_class
        mock_splitter_class.return_value = mock_splitter
        mock_importlib.import_module.return_value = mock_module

        # Mock vector store search results
        mock_docs = [Document(page_content="First document"), Document(page_content="Second document")]
        mock_vector_store.asimilarity_search = AsyncMock(return_value=mock_docs)

        # Create tool
        tool = factory.create_tool(basic_config)

        # Verify tool properties
        assert tool.name == "test_search"
        assert tool.description == "Test search tool"

        # Verify vector store registry was created correctly
        mock_registry_class.create_from_config.assert_called_once_with("test_registry")
        mock_registry.get.assert_called_once()

        # Verify text splitter was created correctly
        mock_importlib.import_module.assert_called_once_with("langchain.text_splitter")
        mock_splitter_class.assert_called_once_with(chunk_size=500)

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    def test_create_tool_vector_store_error(self, mock_registry_class, factory, basic_config):
        """Test error when vector store creation fails."""
        mock_registry_class.create_from_config.side_effect = Exception("Vector store error")

        with pytest.raises(ValueError, match="Failed to create vector store from registry 'test_registry'"):
            factory.create_tool(basic_config)

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    def test_create_tool_text_splitter_import_error(self, mock_importlib, mock_registry_class, factory, basic_config):
        """Test error when text splitter import fails."""
        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = Mock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock import error
        mock_importlib.import_module.side_effect = ImportError("Module not found")

        with pytest.raises(ValueError, match="Failed to import text splitter class 'RecursiveCharacterTextSplitter'"):
            factory.create_tool(basic_config)

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_success(self, mock_importlib, mock_registry_class, factory, basic_config):
        """Test tool ainvoke returns formatted documents."""
        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = AsyncMock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock text splitter
        mock_module = Mock()
        mock_splitter_class = Mock()
        mock_module.RecursiveCharacterTextSplitter = mock_splitter_class
        mock_importlib.import_module.return_value = mock_module

        # Mock search results
        mock_docs = [Document(page_content="First document content"), Document(page_content="Second document content")]
        mock_vector_store.asimilarity_search = AsyncMock(return_value=mock_docs)

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        # Verify search was called correctly
        mock_vector_store.asimilarity_search.assert_called_once_with("test query", k=3)

        # Verify result formatting
        expected_result = "Document 1: First document content\n\nDocument 2: Second document content"
        assert result == expected_result

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_with_filter(self, mock_importlib, mock_registry_class, factory):
        """Test tool ainvoke with filter expression."""
        config = RAGToolConfig(embeddings_store="test_registry", filter_expression={"category": "technical"}, top_k=2)

        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = AsyncMock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock text splitter
        mock_module = Mock()
        mock_splitter_class = Mock()
        mock_module.RecursiveCharacterTextSplitter = mock_splitter_class
        mock_importlib.import_module.return_value = mock_module

        # Mock search results
        mock_docs = [Document(page_content="Filtered document")]
        mock_vector_store.asimilarity_search = AsyncMock(return_value=mock_docs)

        # Create and invoke tool
        tool = factory.create_tool(config)
        result = await tool.ainvoke({"query": "test query"})

        # Verify search was called with filter
        mock_vector_store.asimilarity_search.assert_called_once_with(
            "test query", k=2, filter={"category": "technical"}
        )

        assert result == "Document 1: Filtered document"

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_no_results(self, mock_importlib, mock_registry_class, factory, basic_config):
        """Test tool ainvoke when no documents are found."""
        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = AsyncMock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock text splitter
        mock_module = Mock()
        mock_splitter_class = Mock()
        mock_module.RecursiveCharacterTextSplitter = mock_splitter_class
        mock_importlib.import_module.return_value = mock_module

        # Mock empty search results
        mock_vector_store.asimilarity_search = AsyncMock(return_value=[])

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert result == "No relevant documents found."

    @patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
    @patch("genai_tk.tools.langchain.rag_tool_factory.importlib")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_search_error(self, mock_importlib, mock_registry_class, factory, basic_config):
        """Test tool ainvoke handles search errors gracefully."""
        # Mock vector store registry
        mock_registry = Mock()
        mock_vector_store = AsyncMock()
        mock_registry.get.return_value = mock_vector_store
        mock_registry_class.create_from_config.return_value = mock_registry

        # Mock text splitter
        mock_module = Mock()
        mock_splitter_class = Mock()
        mock_module.RecursiveCharacterTextSplitter = mock_splitter_class
        mock_importlib.import_module.return_value = mock_module

        # Mock search error
        mock_vector_store.asimilarity_search = AsyncMock(side_effect=Exception("Search failed"))

        # Create and invoke tool
        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert "Error searching vector store: Search failed" in result

    def test_create_tool_from_dict(self, factory):
        """Test creating tool from dictionary configuration."""
        config_dict = {
            "embeddings_store": "test_registry",
            "tool_name": "dict_tool",
            "tool_description": "Tool from dict",
            "text_splitter_config": {"class": "RecursiveCharacterTextSplitter", "chunk_size": 800},
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
