from __future__ import annotations

from typing import Protocol

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class SparseEmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse (BM25-style keyword) vector per input text, in the same order."""
        ...

    def embed_query(self, text: str) -> SparseVector:
        """Return a sparse vector for a single query string."""
        ...


class BM25SparseEmbeddingProvider:
    """Sparse keyword embeddings via FastEmbed's BM25 model.

    Runs entirely on CPU with no external service call after the first model download,
    which is what gives exact §-reference and terminology matches (e.g. "Freiberufler" vs
    "Gewerbe") the precision that dense cosine similarity alone can blur.
    """

    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL) -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [self._to_sparse_vector(embedding) for embedding in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVector:
        return self._to_sparse_vector(next(iter(self._model.query_embed(text))))

    @staticmethod
    def _to_sparse_vector(embedding: object) -> SparseVector:
        return SparseVector(
            indices=embedding.indices.tolist(),  # type: ignore[attr-defined]
            values=embedding.values.tolist(),  # type: ignore[attr-defined]
        )
