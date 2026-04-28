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
        config = RAGToolConfig(
            retriever="hybrid_ensemble",
            tool_name="test_tool",
            tool_description="Test description",
            default_filter={"category": "test"},
            top_k=3,
        )
        assert config.retriever == "hybrid_ensemble"
        assert config.tool_name == "test_tool"
        assert config.tool_description == "Test description"
        assert config.default_filter == {"category": "test"}
        assert config.top_k == 3

    def test_default_values(self):
        config = RAGToolConfig(retriever="default")
        assert config.tool_name == "rag_search"
        assert "Search" in config.tool_description
        assert config.default_filter is None
        assert config.top_k == 4

    def test_none_default_filter(self):
        config = RAGToolConfig(retriever="default", default_filter=None)
        assert config.default_filter is None


class TestRAGToolFactory:
    """Test RAGToolFactory functionality."""

    @pytest.fixture
    def mock_llm(self):
        return Mock(spec=BaseChatModel)

    @pytest.fixture
    def factory(self, mock_llm):
        return RAGToolFactory(mock_llm)

    @pytest.fixture
    def basic_config(self):
        return RAGToolConfig(
            retriever="test_retriever",
            tool_name="test_search",
            tool_description="Test search tool",
            top_k=3,
        )

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    def test_create_tool_success(self, mock_factory_cls, factory, basic_config):
        mock_managed = Mock()
        mock_managed.aquery = AsyncMock(
            return_value=[
                Document(page_content="First document"),
                Document(page_content="Second document"),
            ]
        )
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(basic_config)

        assert tool.name == "test_search"
        assert tool.description == "Test search tool"
        mock_factory_cls.create.assert_called_once_with("test_retriever")

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    def test_create_tool_embeddings_store_error(self, mock_factory_cls, factory, basic_config):
        mock_factory_cls.create.side_effect = ValueError("Retriever not found")

        with pytest.raises(ValueError, match="Failed to create retriever 'test_retriever'"):
            factory.create_tool(basic_config)

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_success(self, mock_factory_cls, factory, basic_config):
        mock_managed = Mock()
        mock_docs = [
            Document(page_content="First document content"),
            Document(page_content="Second document content"),
        ]
        mock_managed.aquery = AsyncMock(return_value=mock_docs)
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        mock_managed.aquery.assert_called_once_with("test query", k=3, filter=None)
        assert "Document 1:" in result
        assert "First document content" in result
        assert "Document 2:" in result
        assert "Second document content" in result

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_with_filter(self, mock_factory_cls, factory):
        config = RAGToolConfig(retriever="test_retriever", default_filter={"category": "technical"}, top_k=2)

        mock_managed = Mock()
        mock_managed.aquery = AsyncMock(return_value=[Document(page_content="Filtered document")])
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(config)
        result = await tool.ainvoke({"query": "test query"})

        mock_managed.aquery.assert_called_once_with("test query", k=2, filter={"category": "technical"})
        assert "Document 1:" in result
        assert "Filtered document" in result

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_with_runtime_filter(self, mock_factory_cls, factory, basic_config):
        mock_managed = Mock()
        mock_managed.aquery = AsyncMock(return_value=[Document(page_content="Runtime filtered document")])
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query", "filter": '{"author": "John"}'})

        mock_managed.aquery.assert_called_once_with("test query", k=3, filter={"author": "John"})
        assert "Document 1:" in result

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_no_results(self, mock_factory_cls, factory, basic_config):
        mock_managed = Mock()
        mock_managed.aquery = AsyncMock(return_value=[])
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert result == "No relevant documents found."

    @patch("genai_tk.core.retriever_factory.RetrieverFactory")
    @pytest.mark.asyncio
    async def test_tool_ainvoke_search_error(self, mock_factory_cls, factory, basic_config):
        mock_managed = Mock()
        mock_managed.aquery = AsyncMock(side_effect=Exception("Search failed"))
        mock_factory_cls.create.return_value = mock_managed

        tool = factory.create_tool(basic_config)
        result = await tool.ainvoke({"query": "test query"})

        assert "Error searching knowledge base:" in result
        assert "Search failed" in result

    def test_create_tool_from_dict(self, factory):
        config_dict = {
            "retriever": "test_retriever",
            "tool_name": "dict_tool",
            "tool_description": "Tool from dict",
            "top_k": 4,
        }

        with patch.object(factory, "create_tool") as mock_create_tool:
            factory.create_tool_from_dict(config_dict)

            args, _ = mock_create_tool.call_args
            config = args[0]
            assert isinstance(config, RAGToolConfig)
            assert config.retriever == "test_retriever"
            assert config.tool_name == "dict_tool"
            assert config.top_k == 4


class TestCreateRAGToolFromConfig:
    """Test convenience function create_rag_tool_from_config."""

    @patch("genai_tk.tools.langchain.rag_tool_factory.RAGToolFactory")
    @patch("genai_tk.core.llm_factory.get_llm")
    def test_create_rag_tool_from_config_with_default_llm(self, mock_get_llm, mock_factory_class):
        mock_llm = Mock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm

        mock_factory = Mock()
        mock_tool = Mock()
        mock_factory.create_tool.return_value = mock_tool
        mock_factory_class.return_value = mock_factory

        config = {"retriever": "default"}

        result = create_rag_tool_from_config(config)

        mock_get_llm.assert_called_once()
        mock_factory_class.assert_called_once_with(mock_llm)
        mock_factory.create_tool.assert_called_once()
        assert result == mock_tool

    @patch("genai_tk.tools.langchain.rag_tool_factory.RAGToolFactory")
    def test_create_rag_tool_from_config_with_provided_llm(self, mock_factory_class):
        mock_llm = Mock(spec=BaseChatModel)

        mock_factory = Mock()
        mock_tool = Mock()
        mock_factory.create_tool.return_value = mock_tool
        mock_factory_class.return_value = mock_factory

        config = {"retriever": "default"}

        result = create_rag_tool_from_config(config, llm=mock_llm)

        mock_factory_class.assert_called_once_with(mock_llm)
        mock_factory.create_tool.assert_called_once()
        assert result == mock_tool
