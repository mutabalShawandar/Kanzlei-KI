from dataclasses import dataclass
from typing import Any

import pytest

from rag.retrieval import QdrantRetriever, RetrievedChunk


@dataclass
class FakePoint:
    payload: dict[str, Any]


@dataclass
class FakeQueryResult:
    points: list[FakePoint]


class FakeQdrantClient:
    """Stub Qdrant client: no network or DB calls, returns canned points in given order."""

    def __init__(self, points: list[FakePoint]) -> None:
        self._points = points
        self.last_call: dict[str, Any] = {}

    async def query_points(self, collection_name: str, query: list[float], limit: int) -> FakeQueryResult:
        self.last_call = {
            "collection_name": collection_name,
            "query": query,
            "limit": limit,
        }
        return FakeQueryResult(points=self._points[:limit])


async def stub_embed_fn(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def make_payload(n: int, section: str | None = "S1") -> dict[str, Any]:
    return {
        "text": f"chunk text {n}",
        "source_id": f"source-{n}",
        "title": f"Title {n}",
        "url": f"https://example.com/{n}",
        "section": section,
    }


@pytest.mark.asyncio
async def test_retrieve_returns_top_k_in_order() -> None:
    points = [FakePoint(payload=make_payload(1)), FakePoint(payload=make_payload(2)), FakePoint(payload=make_payload(3))]
    client = FakeQdrantClient(points)
    retriever = QdrantRetriever(client, "kb", stub_embed_fn)  # type: ignore[arg-type]

    result = await retriever.retrieve("some query", top_k=2)

    assert len(result) == 2
    assert result[0].source_id == "source-1"
    assert result[1].source_id == "source-2"
    assert client.last_call["limit"] == 2
    assert client.last_call["collection_name"] == "kb"


@pytest.mark.asyncio
async def test_retrieve_passes_through_metadata() -> None:
    points = [FakePoint(payload=make_payload(1, section="§14 EStG"))]
    client = FakeQdrantClient(points)
    retriever = QdrantRetriever(client, "kb", stub_embed_fn)  # type: ignore[arg-type]

    result = await retriever.retrieve("query", top_k=1)

    chunk = result[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.text == "chunk text 1"
    assert chunk.title == "Title 1"
    assert chunk.url == "https://example.com/1"
    assert chunk.section == "§14 EStG"


@pytest.mark.asyncio
async def test_retrieve_uses_embedding_function() -> None:
    embedded_queries: list[str] = []

    async def tracking_embed_fn(text: str) -> list[float]:
        embedded_queries.append(text)
        return [1.0, 2.0]

    client = FakeQdrantClient([FakePoint(payload=make_payload(1))])
    retriever = QdrantRetriever(client, "kb", tracking_embed_fn)  # type: ignore[arg-type]

    await retriever.retrieve("what is a Freiberufler", top_k=1)

    assert embedded_queries == ["what is a Freiberufler"]
    assert client.last_call["query"] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retrieve_rejects_non_positive_top_k() -> None:
    client = FakeQdrantClient([])
    retriever = QdrantRetriever(client, "kb", stub_embed_fn)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await retriever.retrieve("query", top_k=0)


@pytest.mark.asyncio
async def test_retrieve_skips_points_with_incomplete_payload() -> None:
    points = [
        FakePoint(payload={"text": "incomplete, missing keys"}),
        FakePoint(payload=make_payload(2)),
    ]
    client = FakeQdrantClient(points)
    retriever = QdrantRetriever(client, "kb", stub_embed_fn)  # type: ignore[arg-type]

    result = await retriever.retrieve("query", top_k=2)

    assert len(result) == 1
    assert result[0].source_id == "source-2"
