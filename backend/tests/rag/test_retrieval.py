from dataclasses import dataclass
from typing import Any

import pytest
from qdrant_client.models import SparseVector

from ingest.qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from rag.retrieval import QdrantRetriever, RetrievedChunk


@dataclass
class FakePoint:
    id: str
    payload: dict[str, Any]


@dataclass
class FakeQueryResult:
    points: list[FakePoint]


class FakeQdrantClient:
    """Stub Qdrant client: no network or DB calls, returns canned per-vector-name results."""

    def __init__(self, points_by_vector_name: dict[str, list[FakePoint]]) -> None:
        self._points_by_vector_name = points_by_vector_name
        self.calls: list[dict[str, Any]] = []

    async def query_points(self, collection_name: str, query: Any, using: str, limit: int) -> FakeQueryResult:
        self.calls.append({"collection_name": collection_name, "query": query, "using": using, "limit": limit})
        return FakeQueryResult(points=self._points_by_vector_name.get(using, [])[:limit])


class FakeReranker:
    """Stub reranker: returns candidates unchanged (identity), or driven by an injected scorer."""

    def __init__(self, score_fn=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._score_fn = score_fn or (lambda query, chunk: 0.0)

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "candidates": list(candidates), "top_k": top_k})
        ranked = sorted(candidates, key=lambda c: self._score_fn(query, c), reverse=True)
        return ranked[:top_k]


async def stub_embed_fn(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def stub_sparse_embed_fn(text: str) -> SparseVector:
    return SparseVector(indices=[1, 2], values=[0.5, 0.5])


def make_payload(n: int, section: str | None = "S1") -> dict[str, Any]:
    return {
        "text": f"chunk text {n}",
        "source_id": f"source-{n}",
        "title": f"Title {n}",
        "url": f"https://example.com/{n}",
        "section": section,
    }


def make_retriever(
    points_by_vector_name: dict[str, list[FakePoint]],
    reranker: FakeReranker | None = None,
    candidate_limit: int = 20,
) -> tuple[QdrantRetriever, FakeQdrantClient, FakeReranker]:
    client = FakeQdrantClient(points_by_vector_name)
    reranker = reranker or FakeReranker()
    retriever = QdrantRetriever(
        client,  # type: ignore[arg-type]
        "kb",
        stub_embed_fn,
        stub_sparse_embed_fn,
        reranker,  # type: ignore[arg-type]
        candidate_limit=candidate_limit,
    )
    return retriever, client, reranker


@pytest.mark.asyncio
async def test_retrieve_queries_both_dense_and_sparse_vectors() -> None:
    points = {
        DENSE_VECTOR_NAME: [FakePoint(id="1", payload=make_payload(1))],
        SPARSE_VECTOR_NAME: [FakePoint(id="1", payload=make_payload(1))],
    }
    retriever, client, _ = make_retriever(points)

    await retriever.retrieve("some query", top_k=1)

    used = {call["using"] for call in client.calls}
    assert used == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}
    assert all(call["collection_name"] == "kb" for call in client.calls)


@pytest.mark.asyncio
async def test_retrieve_fuses_dense_and_sparse_hits_agreeing_point_ranks_higher() -> None:
    # Point "2" appears in both rankings; point "1" only in dense, point "3" only in sparse.
    # RRF should place "2" first despite point "1" being the top dense hit alone.
    points = {
        DENSE_VECTOR_NAME: [FakePoint(id="1", payload=make_payload(1)), FakePoint(id="2", payload=make_payload(2))],
        SPARSE_VECTOR_NAME: [FakePoint(id="2", payload=make_payload(2)), FakePoint(id="3", payload=make_payload(3))],
    }
    # Identity-ish reranker: preserve fused order by scoring on input position (no-op passthrough).
    retriever, _, reranker = make_retriever(points, reranker=FakeReranker())

    await retriever.retrieve("query", top_k=3)

    fused_source_ids = [chunk.source_id for chunk in reranker.calls[0]["candidates"]]
    assert fused_source_ids[0] == "source-2"


@pytest.mark.asyncio
async def test_retrieve_passes_wide_candidate_set_to_reranker() -> None:
    dense_points = [FakePoint(id=str(i), payload=make_payload(i)) for i in range(20)]
    points = {DENSE_VECTOR_NAME: dense_points, SPARSE_VECTOR_NAME: []}
    retriever, client, reranker = make_retriever(points, candidate_limit=20)

    await retriever.retrieve("query", top_k=5)

    assert client.calls[0]["limit"] == 20
    assert len(reranker.calls[0]["candidates"]) == 20
    assert reranker.calls[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_retrieve_reranking_can_change_ordering_vs_dense_only_baseline() -> None:
    dense_points = [FakePoint(id="1", payload=make_payload(1)), FakePoint(id="2", payload=make_payload(2))]
    points = {DENSE_VECTOR_NAME: dense_points, SPARSE_VECTOR_NAME: []}

    def prefer_source_2(query: str, chunk: RetrievedChunk) -> float:
        return 1.0 if chunk.source_id == "source-2" else 0.0

    retriever, _, _ = make_retriever(points, reranker=FakeReranker(score_fn=prefer_source_2))

    result = await retriever.retrieve("query", top_k=2)

    assert result[0].source_id == "source-2"  # dense-only baseline would have ranked source-1 first


@pytest.mark.asyncio
async def test_retrieve_passes_through_metadata() -> None:
    points = {
        DENSE_VECTOR_NAME: [FakePoint(id="1", payload=make_payload(1, section="§14 EStG"))],
        SPARSE_VECTOR_NAME: [],
    }
    retriever, _, _ = make_retriever(points)

    result = await retriever.retrieve("query", top_k=1)

    chunk = result[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.text == "chunk text 1"
    assert chunk.title == "Title 1"
    assert chunk.url == "https://example.com/1"
    assert chunk.section == "§14 EStG"


@pytest.mark.asyncio
async def test_retrieve_uses_embedding_functions() -> None:
    embedded_queries: list[str] = []
    sparse_embedded_queries: list[str] = []

    async def tracking_embed_fn(text: str) -> list[float]:
        embedded_queries.append(text)
        return [1.0, 2.0]

    def tracking_sparse_embed_fn(text: str) -> SparseVector:
        sparse_embedded_queries.append(text)
        return SparseVector(indices=[0], values=[1.0])

    points = {DENSE_VECTOR_NAME: [FakePoint(id="1", payload=make_payload(1))], SPARSE_VECTOR_NAME: []}
    client = FakeQdrantClient(points)
    retriever = QdrantRetriever(
        client,  # type: ignore[arg-type]
        "kb",
        tracking_embed_fn,
        tracking_sparse_embed_fn,
        FakeReranker(),  # type: ignore[arg-type]
    )

    await retriever.retrieve("what is a Freiberufler", top_k=1)

    assert embedded_queries == ["what is a Freiberufler"]
    assert sparse_embedded_queries == ["what is a Freiberufler"]
    assert client.calls[0]["query"] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retrieve_rejects_non_positive_top_k() -> None:
    retriever, _, _ = make_retriever({DENSE_VECTOR_NAME: [], SPARSE_VECTOR_NAME: []})

    with pytest.raises(ValueError):
        await retriever.retrieve("query", top_k=0)


@pytest.mark.asyncio
async def test_retrieve_skips_points_with_incomplete_payload() -> None:
    points = {
        DENSE_VECTOR_NAME: [
            FakePoint(id="1", payload={"text": "incomplete, missing keys"}),
            FakePoint(id="2", payload=make_payload(2)),
        ],
        SPARSE_VECTOR_NAME: [],
    }
    retriever, _, reranker = make_retriever(points)

    await retriever.retrieve("query", top_k=2)

    assert len(reranker.calls[0]["candidates"]) == 1
    assert reranker.calls[0]["candidates"][0].source_id == "source-2"
