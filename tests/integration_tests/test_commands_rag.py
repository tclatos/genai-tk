"""Integration tests for RAG CLI commands.

This module tests the CLI commands for vector store manipulation,
including embedding text, querying documents, and managing vector stores.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.extra.rag.commands_rag import register_commands


@pytest.fixture
def cli_app():
    """Create a Typer app with RAG commands registered."""
    app = typer.Typer()
    register_commands(app)
    return app


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


class TestRAGCommands:
    """Test suite for RAG CLI commands."""

    def test_list_configs_command(self, cli_app, cli_runner):
        """Test listing available vector store configurations."""
        result = cli_runner.invoke(cli_app, ["rag", "list-configs"])

        # Should succeed with exit code 0
        assert result.exit_code == 0

        # Should show available configurations from baseline.yaml
        output = result.stdout
        assert "Available Vector Store Configurations" in output

        # Should show column headers for the enhanced table
        if "No Configurations" not in output:
            assert "Backend" in output
            assert "Storage" in output
            assert "Embeddings" in output
            assert "default" in output

    def test_info_command_with_default_config(self, cli_app, cli_runner):
        """Test getting info about the default vector store."""
        result = cli_runner.invoke(cli_app, ["rag", "info", "default"])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show vector store information
        assert "Vector Store Information" in output
        assert "Backend" in output
        assert "InMemory" in output or "Chroma" in output

    def test_info_command_with_invalid_config(self, cli_app, cli_runner):
        """Test getting info about a non-existent vector store."""
        result = cli_runner.invoke(cli_app, ["rag", "info", "nonexistent_store"])

        # Should succeed (error is handled gracefully)
        assert result.exit_code == 0

        output = result.stdout
        # Should show error message
        assert "Configuration Not Found" in output
        assert "nonexistent_store" in output

    def test_embed_command_with_text_option(self, cli_app, cli_runner):
        """Test embedding text using the --text option."""
        test_text = "This is a test document for embedding"

        result = cli_runner.invoke(cli_app, ["rag", "embed", "default", "--text", test_text])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show success message
        assert "Embedding Complete" in output or "Successfully embedded" in output

    def test_embed_command_with_metadata(self, cli_app, cli_runner):
        """Test embedding text with JSON metadata."""
        test_text = "This is a test document with metadata"
        test_metadata = {"source": "test", "category": "example"}

        result = cli_runner.invoke(
            cli_app, ["rag", "embed", "default", "--text", test_text, "--metadata", json.dumps(test_metadata)]
        )

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        assert "Embedding Complete" in output or "Successfully embedded" in output

    def test_embed_command_with_invalid_metadata(self, cli_app, cli_runner):
        """Test embedding text with invalid JSON metadata."""
        test_text = "This is a test document"
        invalid_metadata = "not-valid-json"

        result = cli_runner.invoke(
            cli_app, ["rag", "embed", "default", "--text", test_text, "--metadata", invalid_metadata]
        )

        # Should succeed (error handled gracefully)
        assert result.exit_code == 0

        output = result.stdout
        assert "Invalid Metadata" in output

    def test_embed_command_without_text(self, cli_app, cli_runner):
        """Test embedding command without providing text."""
        result = cli_runner.invoke(cli_app, ["rag", "embed", "default"])

        # Should succeed (error handled gracefully)
        assert result.exit_code == 0

        output = result.stdout
        assert "Empty Input" in output or "No Input" in output

    def test_query_command_basic(self, cli_app, cli_runner):
        """Test querying a vector store."""
        # First embed some text
        cli_runner.invoke(cli_app, ["rag", "embed", "default", "--text", "Python is a programming language"])

        # Then query it
        result = cli_runner.invoke(cli_app, ["rag", "query", "default", "programming language"])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show query results or "No Results"
        assert "Query Results" in output or "No Results" in output

    def test_query_command_with_parameters(self, cli_app, cli_runner):
        """Test querying with k and filter parameters."""
        # First embed some text
        cli_runner.invoke(cli_app, ["rag", "embed", "default", "--text", "Machine learning is a subset of AI"])

        # Query with k parameter
        result = cli_runner.invoke(cli_app, ["rag", "query", "default", "machine learning", "--k", "2"])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show query results
        assert "Query" in output or "No Results" in output or "machine learning" in output

    def test_query_command_invalid_k(self, cli_app, cli_runner):
        """Test query command with invalid k parameter."""
        result = cli_runner.invoke(cli_app, ["rag", "query", "default", "test query", "--k", "0"])

        # Should succeed (error handled gracefully)
        assert result.exit_code == 0

        output = result.stdout
        assert "Invalid Parameter" in output
        assert "k must be at least 1" in output

    def test_query_command_with_filter(self, cli_app, cli_runner):
        """Test query command with metadata filter parameter."""
        # Test with a valid JSON filter
        result = cli_runner.invoke(
            cli_app, ["rag", "query", "default", "test query", "--filter", '{"file_hash": "abc123"}']
        )

        # Should succeed (even if no results found)
        assert result.exit_code == 0

        output = result.stdout
        # Should show query summary or no results
        assert "Query" in output or "No Results" in output

    def test_query_command_invalid_filter(self, cli_app, cli_runner):
        """Test query command with invalid filter JSON."""
        result = cli_runner.invoke(cli_app, ["rag", "query", "default", "test query", "--filter", "not valid json"])

        # Should succeed (error handled gracefully)
        assert result.exit_code == 0

        output = result.stdout
        assert "Invalid Filter" in output or "Failed to parse" in output

    def test_delete_command_empty_store(self, cli_app, cli_runner):
        """Test deleting from an empty vector store."""
        result = cli_runner.invoke(cli_app, ["rag", "delete", "default", "--force"])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show warning or success message
        assert "Nothing to Delete" in output or "Deletion Complete" in output

    def test_delete_command_with_documents(self, cli_app, cli_runner):
        """Test deleting from a vector store with documents."""
        # First embed some documents
        cli_runner.invoke(cli_app, ["rag", "embed", "default", "--text", "Document to be deleted"])

        # Then delete
        result = cli_runner.invoke(cli_app, ["rag", "delete", "default", "--force"])

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        # Should show deletion message
        assert "Deletion Complete" in output or "deleted" in output.lower()

    def test_workflow_embed_query_delete(self, cli_app, cli_runner):
        """Test a complete workflow: embed, query, delete."""
        # Step 1: Embed a document
        embed_result = cli_runner.invoke(
            cli_app,
            [
                "rag",
                "embed",
                "default",
                "--text",
                "Vector databases enable semantic search",
                "--metadata",
                json.dumps({"source": "test_workflow"}),
            ],
        )
        assert embed_result.exit_code == 0

        # Step 2: Query for the document
        query_result = cli_runner.invoke(cli_app, ["rag", "query", "default", "semantic search", "--k", "1"])
        assert query_result.exit_code == 0

        # Step 3: Get info about the store
        info_result = cli_runner.invoke(cli_app, ["rag", "info", "default"])
        assert info_result.exit_code == 0

        # Step 4: Delete all documents
        delete_result = cli_runner.invoke(cli_app, ["rag", "delete", "default"])
        assert delete_result.exit_code == 0

    @pytest.mark.parametrize(
        "invalid_store",
        [
            "nonexistent_store",
            "invalid-store-name",
            "123invalid",
        ],
    )
    def test_commands_with_invalid_store_names(self, cli_app, cli_runner, invalid_store):
        """Test commands with various invalid store names."""
        # Test each command with invalid store name
        commands_to_test = [
            ["rag", "info", invalid_store],
            ["rag", "embed", invalid_store, "--text", "test"],
            ["rag", "query", invalid_store, "test query"],
            ["rag", "delete", invalid_store],
        ]

        for command in commands_to_test:
            result = cli_runner.invoke(cli_app, command)
            # Should handle error gracefully
            assert result.exit_code == 0
            # Should show error message
            output = result.stdout
            assert "Configuration Not Found" in output or "not found" in output.lower()

    def test_embed_command_with_stdin(self, cli_app, cli_runner):
        """Test embedding text from stdin."""
        test_text = "This text comes from stdin"

        result = cli_runner.invoke(cli_app, ["rag", "embed", "default"], input=test_text)

        # Should succeed
        assert result.exit_code == 0

        output = result.stdout
        assert "Embedding Complete" in output or "Successfully embedded" in output


class TestRAGCommandsWithCustomConfig:
    """Test RAG commands with custom vector store configurations."""

    def test_with_in_memory_chroma_config(self, cli_app, cli_runner):
        """Test commands with in-memory Chroma configuration if available."""
        # First check if the config exists
        list_result = cli_runner.invoke(cli_app, ["rag", "list-configs"])
        if "in_memory_chroma" not in list_result.stdout:
            pytest.skip("in_memory_chroma configuration not available")

        # Test info command
        result = cli_runner.invoke(cli_app, ["rag", "info", "in_memory_chroma"])
        assert result.exit_code == 0

        output = result.stdout
        assert "Vector Store Information" in output
        assert "Chroma" in output


@pytest.mark.slow
class TestRAGCommandsIntegration:
    """Slower integration tests that test more complex scenarios."""

    def test_large_text_embedding(self, cli_app, cli_runner):
        """Test embedding of large text documents."""
        large_text = "Large document content. " * 100

        result = cli_runner.invoke(cli_app, ["rag", "embed", "default", "--text", large_text])

        assert result.exit_code == 0

        # Should handle large text gracefully
        output = result.stdout
        assert "Embedding Complete" in output

    def test_multiple_documents_workflow(self, cli_app, cli_runner):
        """Test workflow with multiple documents."""
        documents = [
            ("Python is a programming language", {"type": "programming"}),
            ("Machine learning uses algorithms", {"type": "ai"}),
            ("Databases store structured data", {"type": "data"}),
        ]

        # Embed multiple documents
        for text, metadata in documents:
            result = cli_runner.invoke(
                cli_app, ["rag", "embed", "default", "--text", text, "--metadata", json.dumps(metadata)]
            )
            assert result.exit_code == 0

        # Query should find relevant documents
        query_result = cli_runner.invoke(cli_app, ["rag", "query", "default", "programming algorithms", "--k", "3"])
        assert query_result.exit_code == 0

        # Clean up
        delete_result = cli_runner.invoke(cli_app, ["rag", "delete", "default"])
        assert delete_result.exit_code == 0
