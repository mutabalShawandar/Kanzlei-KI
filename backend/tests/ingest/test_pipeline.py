from datetime import datetime, timezone

import pytest
from qdrant_client import QdrantClient

from ingest.pipeline import EmptyIngestionBatchError, ingest_documents
from ingest.qdrant_store import point_id_for
from ingest.sources.base import SourceDocument

RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeEmbeddingProvider:
    def __init__(self, vector_size: int = 3) -> None:
        self.vector_size = vector_size
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(i)] * self.vector_size for i in range(len(texts))]


def _document(source_id: str, text: str) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_type="statute",
        title="EStG §1",
        url="https://gesetze-im-internet.de/estg/__1.html",
        section="§1",
        text=text,
        retrieved_at=RETRIEVED_AT,
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.mark.asyncio
async def test_ingest_documents_chunks_embeds_and_upserts(client: QdrantClient) -> None:
    documents = [
        _document("doc-1", "Para one.\n\nPara two."),
        _document("doc-2", "Only paragraph."),
    ]
    provider = FakeEmbeddingProvider()

    count = await ingest_documents(documents, "kb", provider, client=client)

    assert count == 2
    assert client.count("kb").count == 2
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 2


@pytest.mark.asyncio
async def test_ingest_documents_is_idempotent_across_runs(client: QdrantClient) -> None:
    documents = [_document("doc-1", "Only paragraph.")]
    provider = FakeEmbeddingProvider()

    await ingest_documents(documents, "kb", provider, client=client)
    await ingest_documents(documents, "kb", provider, client=client)

    assert client.count("kb").count == 1


@pytest.mark.asyncio
async def test_ingest_documents_raises_on_empty_input(client: QdrantClient) -> None:
    provider = FakeEmbeddingProvider()

    with pytest.raises(EmptyIngestionBatchError):
        await ingest_documents([], "kb", provider, client=client)


@pytest.mark.asyncio
async def test_ingest_documents_uses_stable_point_ids(client: QdrantClient) -> None:
    documents = [_document("doc-1", "Only paragraph.")]
    provider = FakeEmbeddingProvider()

    await ingest_documents(documents, "kb", provider, client=client)

    point = client.retrieve("kb", ids=[point_id_for("doc-1", 0)])
    assert len(point) == 1
