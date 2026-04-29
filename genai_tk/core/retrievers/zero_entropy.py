"""ZeroEntropy retriever builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ZeroEntropyRetrieverConfig(BaseModel):
    """Configuration for a ZeroEntropy SDK retriever (read-only, external service)."""

    collection_name: str
    k: int = 5
    retrieval_type: str = "documents"


class ZeroEntropyRetriever:
    """Builder for ZeroEntropy SDK retrievers."""

    config_model = ZeroEntropyRetrieverConfig

    @classmethod
    def build(cls, cfg: ZeroEntropyRetrieverConfig, config_tag: str, resolver: Callable[[str], Any]) -> Any:
        from genai_tk.core.retriever_factory import ManagedRetriever
        from genai_tk.extra.retrievers.zeroentropy import ZeroEntropyConfig
        from genai_tk.extra.retrievers.zeroentropy import ZeroEntropyRetriever as _ZERetriever

        ze_cfg = ZeroEntropyConfig(
            collection_name=cfg.collection_name,
            k=cfg.k,
            retrieval_type=cfg.retrieval_type,
        )
        return ManagedRetriever(
            retriever=_ZERetriever(config=ze_cfg),
            store=None,
            default_k=cfg.k,
            config_tag=config_tag,
        )
