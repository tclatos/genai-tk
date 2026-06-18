"""FastEmbed-backed LangChain Embeddings implementation.

Wraps the ``fastembed`` package directly, providing a ``langchain_core``-compatible
``Embeddings`` class without depending on ``langchain-community`` (which is depreciated).
Check if supporte version exists.

Usage (via config/providers/providers.yaml)::

    local:
      use: genai_tk.utils.langchain_community_repl.fastembed_embeddings.FastEmbedEmbeddings

Direct usage::

    from genai_tk.utils.langchain_community_repl.fastembed_embeddings import FastEmbedEmbeddings

    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectors = emb.embed_documents(["hello", "world"])
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict


class FastEmbedEmbeddings(BaseModel, Embeddings):
    """LangChain Embeddings backed by the ``fastembed`` package.

    Attributes:
        model_name: FastEmbed model identifier (see https://qdrant.github.io/fastembed/).
        cache_dir: Optional directory to cache downloaded ONNX models.
        batch_size: Number of texts to embed per forward pass.
        parallel: Number of parallel workers (0 = auto).
        model_kwargs: Extra kwargs forwarded to ``fastembed.TextEmbedding``.
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    cache_dir: str | None = None
    batch_size: int = 256
    parallel: int | None = None
    model_kwargs: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_model(self):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the 'local' embeddings provider. Install: uv add fastembed"
            ) from exc
        kwargs: dict[str, Any] = {"model_name": self.model_name, **self.model_kwargs}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return TextEmbedding(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._build_model()
        return [list(v) for v in model.embed(texts, batch_size=self.batch_size, parallel=self.parallel)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
