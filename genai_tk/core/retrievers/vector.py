"""Vector-similarity retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class VectorRetrieverConfig(BaseModel):
    """Configuration for a vector-similarity retriever."""

    embeddings_store: str
    top_k: int = 4
    search_type: str = "similarity"
    record_manager_url: str | None = None


class VectorRetriever:
    """Builder for vector-similarity retrievers backed by an EmbeddingsStore."""

    config_model = VectorRetrieverConfig

    @classmethod
    def build(cls, cfg: VectorRetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from genai_tk.core.embeddings_store import EmbeddingsStore
        from genai_tk.core.retriever_factory import (
            ManagedRetriever,
            VectorDocumentStore,
            _is_persistent,
            _make_record_manager,
        )

        es = EmbeddingsStore.create_from_config(cfg.embeddings_store)
        vs = es.get_vector_store()
        rm = _make_record_manager(
            cfg.record_manager_url,
            backend=es.backend or "unknown",
            table_name=es.table_name,
            config_tag=config_tag,
            is_persistent=_is_persistent(es),
        )
        retriever = vs.as_retriever(search_kwargs={"k": cfg.top_k})
        store = VectorDocumentStore(vector_store=vs, record_manager=rm)
        return ManagedRetriever(
            retriever=retriever,
            store=store,
            default_k=cfg.top_k,
            config_tag=config_tag,
            vector_store=vs,
        )
