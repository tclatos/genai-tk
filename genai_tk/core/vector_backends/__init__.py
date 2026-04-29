"""Stable facade for VectorStore backend classes.

Each backend class exposes:
- ``create(embeddings_factory, table_name, config, collection_metadata)`` → ``VectorStore``

YAML usage (``embeddings_store:`` section)::

    embeddings_store:
      my_store:
        backend: genai_tk.core.vector_backends.ChromaBackend    # qualified (preferred)
        embeddings: default
        config:
          storage: '::memory::'

      disk_store:
        backend: Chroma                                          # short alias (backward compat)
        embeddings: default
        config:
          storage: ${paths.data_root}/vector_store

Short aliases::

    Chroma    →  genai_tk.core.vector_backends.ChromaBackend
    InMemory  →  genai_tk.core.vector_backends.InMemoryBackend
    PgVector  →  genai_tk.core.vector_backends.PgVectorBackend
"""

from genai_tk.core.vector_backends.chroma import ChromaBackend
from genai_tk.core.vector_backends.in_memory import InMemoryBackend
from genai_tk.core.vector_backends.pgvector import PgVectorBackend

# Short alias → fully-qualified class name (stable references via this __init__)
ALIASES: dict[str, str] = {
    "Chroma": "genai_tk.core.vector_backends.ChromaBackend",
    "InMemory": "genai_tk.core.vector_backends.InMemoryBackend",
    "PgVector": "genai_tk.core.vector_backends.PgVectorBackend",
}

__all__ = [
    "ChromaBackend",
    "InMemoryBackend",
    "PgVectorBackend",
    "ALIASES",
]
