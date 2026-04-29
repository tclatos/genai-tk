"""InMemory vector store backend."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class InMemoryBackend:
    """Creates an ephemeral in-process InMemoryVectorStore."""

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        table_name: str,
        config: dict[str, Any],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:
        from langchain_core.vectorstores import InMemoryVectorStore

        return InMemoryVectorStore(embedding=embeddings)
