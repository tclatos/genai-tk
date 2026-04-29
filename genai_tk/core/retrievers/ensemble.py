"""Ensemble retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class EnsembleChildConfig(BaseModel):
    """One child retriever inside an ensemble."""

    ref: str
    weight: float = 0.5


class EnsembleRetrieverConfig(BaseModel):
    """Configuration for a weighted ensemble of child retrievers."""

    retrievers: list[EnsembleChildConfig]


class EnsembleRetriever:
    """Builder for weighted ensemble retrievers (recursive composition)."""

    config_model = EnsembleRetrieverConfig

    @classmethod
    def build(cls, cfg: EnsembleRetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from langchain_classic.retrievers import EnsembleRetriever as LangchainEnsembleRetriever

        from genai_tk.core.retriever_factory import CompositeDocumentStore, ManagedRetriever

        children = [resolver(child.ref) for child in cfg.retrievers]
        weights = [child.weight for child in cfg.retrievers]
        total = sum(weights) or 1.0
        normalised = [w / total for w in weights]

        ensemble = LangchainEnsembleRetriever(
            retrievers=[c.retriever for c in children],
            weights=normalised,
        )
        stores = [c.store for c in children if c.store is not None]
        composite = CompositeDocumentStore(stores=stores) if stores else None
        return ManagedRetriever(
            retriever=ensemble,
            store=composite,
            default_k=4,
            config_tag=config_tag,
        )
