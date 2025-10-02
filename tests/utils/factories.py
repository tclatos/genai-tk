"""Factory classes for generating test data.

This module provides factory classes for creating consistent test data
and mock responses for testing with fake models.
"""

from typing import Any, Dict, List

from langchain_core.documents import Document


class FakeLLMResponseFactory:
    """Factory for creating predictable fake LLM responses."""

    @staticmethod
    def create_joke_response() -> str:
        """Create a predictable joke response."""
        return "Why did the chicken cross the road? To get to the other side!"

    @staticmethod
    def create_summary_response(text: str) -> str:
        """Create a predictable summary response."""
        return f"This is a summary of: {text[:50]}..."

    @staticmethod
    def create_json_response() -> Dict[str, Any]:
        """Create a predictable JSON response."""
        return {
            "status": "success",
            "data": [1, 2, 3],
            "message": "Operation completed successfully",
        }

    @staticmethod
    def create_code_response() -> str:
        """Create a predictable code response."""
        return """
def hello_world():
    print("Hello, World!")
    return "success"
"""

    @staticmethod
    def create_analysis_response() -> str:
        """Create a predictable analysis response."""
        return """
Based on the analysis:
- Key finding 1: The data shows a positive trend
- Key finding 2: Performance metrics are within expected ranges
- Recommendation: Continue with current approach
"""


class FakeTestDataFactory:
    """Factory for creating test data for various components."""

    @staticmethod
    def create_sample_documents(count: int = 5) -> List[Document]:
        """Create sample documents for testing."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a powerful programming language for AI development.",
            "Machine learning models require training data to perform well.",
            "Vector databases enable efficient similarity search.",
            "Embeddings represent text as numerical vectors.",
            "Artificial intelligence is transforming many industries.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning learns through trial and error.",
        ]

        return [
            Document(page_content=texts[i % len(texts)], metadata={"id": i, "source": f"doc_{i}", "page": i // 2 + 1})
            for i in range(count)
        ]

    @staticmethod
    def create_sample_texts(count: int = 5) -> List[str]:
        """Create sample text strings for testing."""
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a powerful programming language for AI development.",
            "Machine learning models require training data to perform well.",
            "Vector databases enable efficient similarity search.",
            "Embeddings represent text as numerical vectors.",
        ]

        return [base_texts[i % len(base_texts)] for i in range(count)]

    @staticmethod
    def create_sample_queries() -> List[str]:
        """Create sample search queries for testing."""
        return [
            "programming language",
            "machine learning",
            "similarity search",
            "artificial intelligence",
            "data science",
            "neural networks",
            "computer vision",
        ]

    @staticmethod
    def create_test_config() -> Dict[str, Any]:
        """Create a test configuration dictionary."""
        return {
            "llm": {
                "models": {
                    "default": "parrot_local_fake",
                    "fake": "parrot_local_fake",
                },
                "cache": "memory",
            },
            "embeddings": {
                "models": {
                    "default": "embeddings_768_fake",
                    "fake": "embeddings_768_fake",
                },
                "cache": True,
            },
            "vector_store": {
                "default": "InMemory",
            },
            "kv_store": {
                "engine": "memory",
            },
        }

    @staticmethod
    def create_tool_definitions() -> List[Dict[str, Any]]:
        """Create sample tool definitions for testing."""
        return [
            {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
            },
        ]
