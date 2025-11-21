import os
import tempfile
from typing import Iterator, List

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel
from upath import UPath


class MarkdownLoader(BaseLoader, BaseModel):
    """Load and chunk Markdown files using Chonkie."""

    file_paths: List[str | UPath]
    chunk_size: int = 500
    chunk_overlap: int = 50
    include_metadata: bool = True
    kwargs: dict = {}

    def model_post_init(self, __context: dict) -> None:
        """Post-initialization logic for validation and setup."""
        # Convert string paths to Path objects
        self.file_paths = [UPath(fp) if isinstance(fp, str) else fp for fp in self.file_paths]

        # Validate file existence and warn about non-Markdown files
        for fp in self.file_paths:
            if not fp.exists():
                raise FileNotFoundError(f"File not found: {fp}")
            if fp.suffix.lower() != ".md":
                logger.warning(f"File {fp} does not have .md extension")

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load and chunk all Markdown files."""
        for file_path in self.file_paths:
            try:
                content = UPath(file_path).read_text()
                chunks = self._chunk_markdown(content, str(file_path))
                for chunk in chunks:
                    yield chunk

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

    def _chunk_markdown(self, content: str, source: str) -> List[Document]:
        """Chunk Markdown content using Chonkie."""

        from chonkie.chef.markdown import MarkdownChef
        from chonkie.chunker.table import TableChunker

        try:
            chef = MarkdownChef()
            table_chunker = TableChunker(chunk_size=self.chunk_size, **self.kwargs)
            processed_content = chef.process(content)
            chunks = table_chunker.chunk(processed_content)

            # Convert to LangChain Documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk.text, metadata={"source": source, "chunk_index": i, "total_chunks": len(chunks)}
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error chunking content from {source}: {e}")
            # Fallback: return single document with full content
            return [Document(page_content=content, metadata={"source": source, "chunk_index": 0})]


def main():
    """Main function demonstrating usage."""
    print("Markdown Loader Demo")
    print("=" * 20)

    # Create a temporary markdown file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\n")
        f.write("This is a test document for the Markdown loader.\n\n")
        f.write("## Section 1\n\n")
        f.write("Content in section 1.\n\n")
        f.write("## Section 2\n\n")
        f.write("Content in section 2 with some additional text to make it longer for chunking purposes.\n\n")
        f.write("### Subsection\n\n")
        f.write("More content here to ensure we have enough text for chunking.\n\n")
        f.write("## Conclusion\n\n")
        f.write("This concludes our test document.\n")
        test_file_path = f.name

    try:
        # Create loader instance
        loader = MarkdownLoader(
            file_paths=[test_file_path],
            chunk_size=100,  # Smaller chunk size for demo
            chunk_overlap=20,
        )

        print(f"Loading and chunking file: {test_file_path}")
        print(f"Chunk size: {loader.chunk_size}, Overlap: {loader.chunk_overlap}")

        # Lazy load documents
        documents = list(loader.lazy_load())

        print(f"\nGenerated {len(documents)} chunks:")
        for i, doc in enumerate(documents):
            print(f"\nChunk {i + 1}:")
            print(f"  Content length: {len(doc.page_content)} chars")
            print(f"  Metadata: source='{doc.metadata['source']}', index={doc.metadata['chunk_index']}")
            print(f"  Content preview: {doc.page_content[:50]}...")

        # Show that regular load() also works
        documents_regular = loader.load()
        print(f"\nRegular load() also produced {len(documents_regular)} documents")

    finally:
        # Clean up temporary file
        os.unlink(test_file_path)


if __name__ == "__main__":
    main()
