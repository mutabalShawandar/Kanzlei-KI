from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel
from qdrant_client import QdrantClient


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
    """Retriever backed by a Qdrant collection and a pluggable embedding function."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embed_fn: Callable[[str], Awaitable[list[float]]],
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embed_fn = embed_fn

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        query_vector = await self._embed_fn(query)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [self._to_chunk(point.payload or {}) for point in results]

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> RetrievedChunk:
        return RetrievedChunk(
            text=payload["text"],
            source_id=payload["source_id"],
            title=payload["title"],
            url=payload["url"],
            section=payload.get("section"),
        )
