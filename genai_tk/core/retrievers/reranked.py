"""Reranked retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class RerankedRetrieverConfig(BaseModel):
    """Configuration for a retriever wrapped with a reranking step."""

    retriever: str
    reranker: str = "embeddings"
    top_k: int = 3
    fetch_k: int = 10
    reranker_model: str | None = None
    embeddings: str | None = None


class RerankedRetriever:
    """Builder for contextual-compression / reranked retrievers."""

    config_model = RerankedRetrieverConfig

    @classmethod
    def build(cls, cfg: RerankedRetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from langchain_classic.retrievers import ContextualCompressionRetriever

        from genai_tk.core.retriever_factory import ManagedRetriever

        base = resolver(cfg.retriever)
        compressor = _build_compressor(cfg)
        reranked = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base.retriever,
        )
        return ManagedRetriever(
            retriever=reranked,
            store=base.store,
            default_k=cfg.top_k,
            config_tag=config_tag,
            vector_store=base._vector_store,
            bm25_store=base._bm25_store,
        )


def _build_compressor(cfg: RerankedRetrieverConfig) -> Any:
    """Build a LangChain document compressor from a RerankedRetrieverConfig."""
    if cfg.reranker == "cohere":
        try:
            from langchain_cohere import CohereRerank  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("langchain-cohere required. Install: uv add langchain-cohere") from exc
        return CohereRerank(model=cfg.reranker_model or "rerank-english-v3.0", top_n=cfg.top_k)

    if cfg.reranker == "cross_encoder":
        try:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "langchain-community required for cross-encoder. Install: uv add langchain-community"
            ) from exc
        model = HuggingFaceCrossEncoder(model_name=cfg.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoderReranker(model=model, top_n=cfg.top_k)

    if cfg.reranker == "embeddings":
        from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

        from genai_tk.core.embeddings_factory import EmbeddingsFactory

        ef = EmbeddingsFactory(embeddings=cfg.embeddings) if cfg.embeddings else EmbeddingsFactory()
        return EmbeddingsFilter(embeddings=ef.get(), similarity_threshold=0.7)

    raise ValueError(f"Unknown reranker '{cfg.reranker}'. Choose from: 'embeddings', 'cohere', 'cross_encoder'.")
