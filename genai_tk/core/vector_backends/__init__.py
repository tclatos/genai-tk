"""Vector backend facade for YAML configuration.

Use ``InMemoryBackend`` for ephemeral work, ``ZvecBackend`` for a persistent local
index, or ``PgVectorBackend`` for PostgreSQL. ``ChromaBackend`` remains importable
only to provide an explicit migration error for legacy configurations.
"""

from genai_tk.core.vector_backends.chroma import ChromaBackend
from genai_tk.core.vector_backends.in_memory import InMemoryBackend
from genai_tk.core.vector_backends.pgvector import PgVectorBackend
from genai_tk.core.vector_backends.zvec import ZvecBackend

__all__ = ["ChromaBackend", "InMemoryBackend", "PgVectorBackend", "ZvecBackend"]
