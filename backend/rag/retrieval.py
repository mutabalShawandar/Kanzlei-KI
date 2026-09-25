from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

from ingest.qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

if TYPE_CHECKING:
    from rag.reranker import Reranker

_REQUIRED_PAYLOAD_KEYS = {"text", "source_id", "title", "url"}

DEFAULT_CANDIDATE_LIMIT = 20
_RRF_K = 60


class RetrievedChunk(BaseModel):
    """Minimal local retrieval result, decoupled from the in-progress ingest.Chunk model.

    Session 2.3 reconciles this with `backend/ingest/chunking.py`'s `Chunk` once merged.
    """

    text: str
    source_id: str
    title: str
    url: str
    section: str | None = None


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks most relevant to query, best match first."""
        ...


class QdrantRetriever:
    """Hybrid (dense + sparse) retriever backed by Qdrant, with cross-encoder reranking.

    Dense and sparse candidates are queried separately and fused with Reciprocal Rank Fusion
    (RRF) rather than Qdrant's server-side fusion query API, so fusion behavior is a plain,
    unit-testable Python function independent of server version/config — important for a
    module this safety-critical. The widened candidate set (`candidate_limit`, default 20) is
    then reranked by a cross-encoder to produce the final top_k, since RRF over independent
    dense/sparse rankings is a coarse relevance signal on its own.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        sparse_embed_fn: Callable[[str], SparseVector],
        reranker: Reranker,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embed_fn = embed_fn
        self._sparse_embed_fn = sparse_embed_fn
        self._reranker = reranker
        self._candidate_limit = candidate_limit

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        dense_vector = await self._embed_fn(query)
        sparse_vector = self._sparse_embed_fn(query)

        dense_response = await self._client.query_points(
            collection_name=self._collection_name,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=self._candidate_limit,
        )
        sparse_response = await self._client.query_points(
            collection_name=self._collection_name,
            query=sparse_vector,
            using=SPARSE_VECTOR_NAME,
            limit=self._candidate_limit,
        )

        fused_points = _reciprocal_rank_fusion(dense_response.points, sparse_response.points)

        candidates = []
        for point in fused_points[: self._candidate_limit]:
            chunk = self._to_chunk(point.payload or {})
            if chunk is not None:
                candidates.append(chunk)

        return await self._reranker.rerank(query, candidates, top_k)

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> RetrievedChunk | None:
        if not _REQUIRED_PAYLOAD_KEYS.issubset(payload):
            return None
        return RetrievedChunk(
            text=payload["text"],
            source_id=payload["source_id"],
            title=payload["title"],
            url=payload["url"],
            section=payload.get("section"),
        )


def _reciprocal_rank_fusion(dense_points: list[Any], sparse_points: list[Any], k: int = _RRF_K) -> list[Any]:
    """Merge two independently-ranked point lists into one ranking via Reciprocal Rank Fusion.

    score(point) = sum over the rankers it appears in of 1 / (k + rank), rank starting at 1.
    A point returned by both dense and sparse search accumulates score from both, so it tends
    to outrank a point that only one signal found.
    """
    scores: dict[str, float] = {}
    points_by_id: dict[str, Any] = {}
    for points in (dense_points, sparse_points):
        for rank, point in enumerate(points, start=1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
            points_by_id[point_id] = point

    ranked_ids = sorted(scores, key=lambda point_id: scores[point_id], reverse=True)
    return [points_by_id[point_id] for point_id in ranked_ids]
