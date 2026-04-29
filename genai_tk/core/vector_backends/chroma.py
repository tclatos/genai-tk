"""Chroma vector store backend."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class ChromaBackend:
    """Creates a Chroma VectorStore (in-memory or persistent)."""

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        table_name: str,
        config: dict[str, Any],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:
        from langchain_chroma import Chroma

        from genai_tk.utils.config_mngr import global_config

        storage = config.get("storage", "::memory::")
        persist_directory: str | None

        if storage == "::memory::":
            persist_directory = None
        elif os.path.isabs(storage):
            persist_directory = storage
        else:
            try:
                persist_directory = str(global_config().get_dir_path("vector_store.storage", create_if_not_exists=True))
            except (ValueError, KeyError):
                persist_directory = storage

        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=table_name,
            collection_metadata=collection_metadata,
        )
