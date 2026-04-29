"""BM25 full-text retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class BM25RetrieverConfig(BaseModel):
    """Configuration for a BM25 full-text retriever."""

    k: int = 4
    preprocessing: str = "default"
    spacy_model: str = "en_core_web_sm"
    cache_dir: str | None = None
    bm25_params: dict[str, Any] = Field(default_factory=dict)


class BM25Retriever:
    """Builder for BM25 full-text retrievers persisted to disk."""

    config_model = BM25RetrieverConfig

    @classmethod
    def build(cls, cfg: BM25RetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from genai_tk.core.retriever_factory import (
            BM25DocumentStore,
            ManagedRetriever,
            _bm25_cache_dir,
            _EmptyRetriever,
        )

        cache_dir = _bm25_cache_dir(cfg.cache_dir, config_tag)
        bm25_store = BM25DocumentStore(
            cache_dir=cache_dir,
            preprocessing=cfg.preprocessing,
            spacy_model=cfg.spacy_model,
            bm25_params=cfg.bm25_params,
        )
        retriever = bm25_store.get_or_load_retriever(k=cfg.k) or _EmptyRetriever()
        return ManagedRetriever(
            retriever=retriever,
            store=bm25_store,
            default_k=cfg.k,
            config_tag=config_tag,
            bm25_store=bm25_store,
        )
