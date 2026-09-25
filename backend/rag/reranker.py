from __future__ import annotations

import asyncio
from typing import Protocol

from sentence_transformers import CrossEncoder

from rag.retrieval import RetrievedChunk

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Re-score candidates against query and return the best top_k, best match first."""
        ...


class CrossEncoderReranker:
    """Reranks hybrid-search candidates with a small multilingual cross-encoder.

    Model choice: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~110M params, 12-layer MiniLM).
    Trained on mMARCO, a machine-translated multilingual passage-ranking dataset that includes
    German, so it scores German query/passage pairs directly rather than relying on cross-lingual
    transfer from an English-only cross-encoder. Tradeoffs: a MiniLM-scale encoder is less
    accurate than a full-size multilingual reranker (e.g. bge-reranker-v2-m3), but it runs
    comfortably on CPU for the ~20-candidate batches this pipeline reranks, which a larger model
    would not do at interactive latency without a GPU. Revisit if Session 1.4's eval harness
    shows recall@k gains from a bigger reranker outweigh the added latency.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
