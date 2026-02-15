"""Shared test data generators for consistent testing.

This module provides reusable test data factories to ensure consistent
test data across all test modules.
"""

from pathlib import Path
from typing import Generator

import pytest
from langchain_core.documents import Document


def generate_sample_documents(count: int = 5) -> list[Document]:
    """Generate sample documents for testing.

    Args:
        count: Number of documents to generate

    Returns:
        List of Document objects with varied content
    """
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for AI development.",
        "Machine learning models require training data to perform well.",
        "Vector databases enable efficient similarity search.",
        "Embeddings represent text as numerical vectors.",
        "Natural language processing transforms human communication.",
        "Deep learning uses neural networks for pattern recognition.",
        "Artificial intelligence mimics human cognitive functions.",
        "Data science combines statistics and programming.",
        "Cloud computing provides scalable infrastructure.",
    ]

    documents = []
    for i in range(min(count, len(base_texts))):
        documents.append(
            Document(
                page_content=base_texts[i],
                metadata={"id": i, "source": f"doc_{i}", "page": i // 2 + 1, "category": "test"},
            )
        )

    # If more documents requested than we have texts, repeat with variation
    if count > len(base_texts):
        for i in range(len(base_texts), count):
            idx = i % len(base_texts)
            documents.append(
                Document(
                    page_content=f"{base_texts[idx]} (variation {i})",
                    metadata={"id": i, "source": f"doc_{i}", "page": i // 2 + 1, "category": "test"},
                )
            )

    return documents


def generate_sample_texts(count: int = 5) -> list[str]:
    """Generate sample text strings for testing.

    Args:
        count: Number of texts to generate

    Returns:
        List of text strings
    """
    return [doc.page_content for doc in generate_sample_documents(count)]


def generate_sample_queries() -> list[str]:
    """Generate sample search queries for testing.

    Returns:
        List of search query strings
    """
    return [
        "programming language",
        "machine learning",
        "similarity search",
        "artificial intelligence",
        "data science",
        "neural networks",
        "vector representation",
        "cloud infrastructure",
    ]


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to temporary test directory
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    yield test_dir


def create_test_file(directory: Path, filename: str, content: str | bytes) -> Path:
    """Create a test file with given content.

    Args:
        directory: Directory to create file in
        filename: Name of the file
        content: Content to write (string or bytes)

    Returns:
        Path to created file
    """
    file_path = directory / filename
    if isinstance(content, str):
        file_path.write_text(content, encoding="utf-8")
    else:
        file_path.write_bytes(content)
    return file_path


def create_sample_text_files(directory: Path, count: int = 3) -> list[Path]:
    """Create sample text files for testing.

    Args:
        directory: Directory to create files in
        count: Number of files to create

    Returns:
        List of paths to created files
    """
    texts = generate_sample_texts(count)
    files = []
    for i, text in enumerate(texts):
        file_path = create_test_file(directory, f"sample_{i}.txt", text)
        files.append(file_path)
    return files


def create_sample_markdown_files(directory: Path, count: int = 3) -> list[Path]:
    """Create sample Markdown files for testing.

    Args:
        directory: Directory to create files in
        count: Number of files to create

    Returns:
        List of paths to created files
    """
    texts = generate_sample_texts(count)
    files = []
    for i, text in enumerate(texts):
        markdown_content = f"# Document {i}\n\n{text}\n\n## Details\n\nThis is a test document.\n"
        file_path = create_test_file(directory, f"sample_{i}.md", markdown_content)
        files.append(file_path)
    return files


def create_sample_json_files(directory: Path, count: int = 3) -> list[Path]:
    """Create sample JSON files for testing.

    Args:
        directory: Directory to create files in
        count: Number of files to create

    Returns:
        List of paths to created files
    """
    import json

    documents = generate_sample_documents(count)
    files = []
    for i, doc in enumerate(documents):
        json_content = json.dumps(
            {"id": i, "content": doc.page_content, "metadata": doc.metadata}, indent=2, ensure_ascii=False
        )
        file_path = create_test_file(directory, f"sample_{i}.json", json_content)
        files.append(file_path)
    return files
