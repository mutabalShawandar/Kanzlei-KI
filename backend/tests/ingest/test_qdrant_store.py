from datetime import datetime, timezone

import pytest
from qdrant_client import QdrantClient

from ingest import qdrant_store
from ingest.chunking import Chunk
from ingest.qdrant_store import ChunkVectorCountMismatchError, ensure_collection, point_id_for, upsert_chunks

RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def _chunk(chunk_index: int, source_id: str = "estg-1") -> Chunk:
    return Chunk(
        source_id=source_id,
        source_type="statute",
        title="EStG §1",
        url="https://gesetze-im-internet.de/estg/__1.html",
        section="§1",
        retrieved_at=RETRIEVED_AT,
        chunk_index=chunk_index,
        text=f"Chunk text {chunk_index}",
    )


def test_ensure_collection_creates_when_missing(client: QdrantClient) -> None:
    ensure_collection(client, "kb", vector_size=4)

    assert client.collection_exists("kb")


def test_ensure_collection_is_idempotent(client: QdrantClient) -> None:
    ensure_collection(client, "kb", vector_size=4)
    ensure_collection(client, "kb", vector_size=4)

    assert client.collection_exists("kb")


def test_upsert_chunks_stores_payload(client: QdrantClient) -> None:
    ensure_collection(client, "kb", vector_size=3)
    chunks = [_chunk(0), _chunk(1)]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    upsert_chunks(client, "kb", chunks, vectors)

    assert client.count("kb").count == 2
    point = client.retrieve("kb", ids=[point_id_for("estg-1", 0)])[0]
    assert point.payload["text"] == "Chunk text 0"
    assert point.payload["source_id"] == "estg-1"


def test_upsert_chunks_is_idempotent_on_re_ingestion(client: QdrantClient) -> None:
    ensure_collection(client, "kb", vector_size=3)
    chunks = [_chunk(0)]

    upsert_chunks(client, "kb", chunks, [[0.1, 0.2, 0.3]])
    upsert_chunks(client, "kb", chunks, [[0.9, 0.9, 0.9]])

    assert client.count("kb").count == 1
    point = client.retrieve("kb", ids=[point_id_for("estg-1", 0)], with_vectors=True)[0]
    # Cosine-distance collections store normalized vectors, so compare direction, not raw values.
    assert point.vector[0] == point.vector[1] == point.vector[2] > 0


def test_point_id_is_stable_for_same_source_and_index() -> None:
    assert point_id_for("estg-1", 0) == point_id_for("estg-1", 0)
    assert point_id_for("estg-1", 0) != point_id_for("estg-1", 1)
    assert point_id_for("estg-1", 0) != point_id_for("estg-2", 0)


def test_upsert_chunks_raises_on_count_mismatch(client: QdrantClient) -> None:
    ensure_collection(client, "kb", vector_size=3)

    with pytest.raises(ChunkVectorCountMismatchError):
        upsert_chunks(client, "kb", [_chunk(0), _chunk(1)], [[0.1, 0.2, 0.3]])


def test_get_client_reads_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    client = qdrant_store.get_client()

    assert client is not None
