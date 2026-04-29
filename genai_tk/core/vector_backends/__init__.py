"""Stable facade for VectorStore backend classes.

Each backend class exposes:
- ``create(embeddings_factory, table_name, config, collection_metadata)`` → ``VectorStore``

YAML usage (``embeddings_store:`` section)::

    embeddings_store:
      my_store:
        backend: genai_tk.core.vector_backends.ChromaBackend
        embeddings: default
        config:
          storage: '::memory::'

      disk_store:
        backend: genai_tk.core.vector_backends.ChromaBackend
        embeddings: default
        config:
          storage: ${paths.data_root}/vector_store
"""

from genai_tk.core.vector_backends.chroma import ChromaBackend
from genai_tk.core.vector_backends.in_memory import InMemoryBackend
from genai_tk.core.vector_backends.pgvector import PgVectorBackend

__all__ = [
    "ChromaBackend",
    "InMemoryBackend",
    "PgVectorBackend",
]
