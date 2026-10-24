"""Migration errors for the retired Chroma vector-store integration."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

CHROMA_MIGRATION_MESSAGE = (
    "Chroma support has been retired because no patched release is available for the reviewed security advisories. "
    "Existing Chroma data has not been opened or changed. Use InMemoryBackend for ephemeral work, "
    "or configure ZvecBackend/PgVectorBackend with a fresh index and re-ingest trusted source documents. "
    "Use the new persistent_zvec or local_zvec examples; do not reuse a Chroma storage directory. "
    "See docs/rag.md#retiring-chroma-and-migrating-existing-indexes."
)

_RETIRED_BACKENDS = {
    "Chroma",
    "chroma",
    "genai_tk.core.vector_backends.ChromaBackend",
    "genai_tk.core.vector_backends.chroma.ChromaBackend",
}


def reject_retired_chroma(backend: str | None) -> None:
    """Reject a legacy backend before resolving models or touching storage."""
    if backend in _RETIRED_BACKENDS:
        raise RuntimeError(CHROMA_MIGRATION_MESSAGE)


class ChromaBackend:
    """Retain an importable migration error for existing Chroma configurations."""

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        table_name: str,
        config: dict[str, object],
        collection_metadata: dict[str, str] | None = None,
    ) -> VectorStore:
        raise RuntimeError(CHROMA_MIGRATION_MESSAGE)
