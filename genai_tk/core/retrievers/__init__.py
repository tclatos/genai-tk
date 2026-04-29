"""Stable facade for retriever builder classes.

Each builder class exposes:
- ``config_model`` — Pydantic model used to parse the YAML sub-dict (without ``type``).
- ``build(cfg, config_tag, resolver)`` — returns a :class:`ManagedRetriever`.

YAML usage::

    retrievers:
      default:
        type: genai_tk.core.retrievers.VectorRetriever    # stable qualified name
        embeddings_store: in_memory_chroma
        top_k: 4

      bm25_local:
        type: genai_tk.core.retrievers.BM25Retriever
        k: 4

Short aliases are supported for backward compatibility::

    type: vector        →  genai_tk.core.retrievers.VectorRetriever
    type: bm25          →  genai_tk.core.retrievers.BM25Retriever
    type: pg_hybrid     →  genai_tk.core.retrievers.PgHybridRetriever
    type: ensemble      →  genai_tk.core.retrievers.EnsembleRetriever
    type: reranked      →  genai_tk.core.retrievers.RerankedRetriever
    type: zero_entropy  →  genai_tk.core.retrievers.ZeroEntropyRetriever
"""

from genai_tk.core.retrievers.bm25 import BM25Retriever
from genai_tk.core.retrievers.ensemble import EnsembleRetriever
from genai_tk.core.retrievers.pg_hybrid import PgHybridRetriever
from genai_tk.core.retrievers.reranked import RerankedRetriever
from genai_tk.core.retrievers.vector import VectorRetriever
from genai_tk.core.retrievers.zero_entropy import ZeroEntropyRetriever

# Short alias → fully-qualified class name (stable references via this __init__)
ALIASES: dict[str, str] = {
    "vector": "genai_tk.core.retrievers.VectorRetriever",
    "bm25": "genai_tk.core.retrievers.BM25Retriever",
    "pg_hybrid": "genai_tk.core.retrievers.PgHybridRetriever",
    "ensemble": "genai_tk.core.retrievers.EnsembleRetriever",
    "reranked": "genai_tk.core.retrievers.RerankedRetriever",
    "zero_entropy": "genai_tk.core.retrievers.ZeroEntropyRetriever",
}

__all__ = [
    "VectorRetriever",
    "BM25Retriever",
    "PgHybridRetriever",
    "EnsembleRetriever",
    "RerankedRetriever",
    "ZeroEntropyRetriever",
    "ALIASES",
]
