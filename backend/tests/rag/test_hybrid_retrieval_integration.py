from datetime import datetime, timezone
from typing import Any

import pytest
from qdrant_client import QdrantClient

from ingest.chunking import Chunk
from ingest.qdrant_store import ensure_collection, upsert_chunks
from ingest.sparse_embedding import BM25SparseEmbeddingProvider
from rag.reranker import CrossEncoderReranker
from rag.retrieval import QdrantRetriever

RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
COLLECTION = "kb"

# Deterministic, identical dense vectors for every chunk: dense search alone cannot distinguish
# them, so any ordering difference in the assertions below must come from sparse/rerank signal.
_UNIFORM_DENSE_VECTOR = [1.0, 0.0, 0.0, 0.0]


class SyncQdrantAsyncAdapter:
    """Wraps a sync in-memory QdrantClient so QdrantRetriever's async query_points calls work.

    QdrantClient(":memory:") and AsyncQdrantClient(":memory:") each spin up their own separate
    embedded backend, so a test can't seed one and query the other. Since the in-memory backend
    has no real I/O, calling the sync method directly (no thread hop) is fine for tests.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    async def query_points(self, collection_name: str, query: Any, using: str, limit: int) -> Any:
        return self._client.query_points(collection_name=collection_name, query=query, using=using, limit=limit)


def _chunk(source_id: str, text: str, index: int = 0) -> Chunk:
    return Chunk(
        source_id=source_id,
        source_type="statute",
        title=source_id,
        url=f"https://example.com/{source_id}",
        section=None,
        retrieved_at=RETRIEVED_AT,
        chunk_index=index,
        text=text,
    )


async def stub_dense_embed_fn(text: str) -> list[float]:
    return _UNIFORM_DENSE_VECTOR


@pytest.mark.asyncio
async def test_hybrid_retrieve_end_to_end_reranking_beats_dense_only_baseline() -> None:
    sparse_provider = BM25SparseEmbeddingProvider()
    sync_client = QdrantClient(":memory:")
    ensure_collection(sync_client, COLLECTION, vector_size=4)

    chunks = [
        _chunk(
            "freiberufler-anmeldung",
            "Freiberufler müssen sich beim Finanzamt mit dem Fragebogen zur steuerlichen "
            "Erfassung anmelden.",
        ),
        _chunk(
            "gbr-gewerbe",
            "Eine GbR, die ein Gewerbe betreibt, muss ein Gewerbe beim Gewerbeamt anmelden.",
        ),
        _chunk(
            "unrelated",
            "Der Bundeshaushalt für Infrastrukturprojekte wurde im letzten Quartal erhöht.",
        ),
    ]
    dense_vectors = [_UNIFORM_DENSE_VECTOR] * len(chunks)
    sparse_vectors = sparse_provider.embed([chunk.text for chunk in chunks])
    upsert_chunks(sync_client, COLLECTION, chunks, dense_vectors, sparse_vectors)

    retriever = QdrantRetriever(
        SyncQdrantAsyncAdapter(sync_client),  # type: ignore[arg-type]
        COLLECTION,
        stub_dense_embed_fn,
        sparse_provider.embed_query,
        CrossEncoderReranker(),
        candidate_limit=20,
    )

    result = await retriever.retrieve("Wie melde ich mich als Freiberufler beim Finanzamt an?", top_k=2)

    # Dense vectors are identical for every chunk, so a dense-only baseline would return them in
    # insertion order (freiberufler-anmeldung first by coincidence here, so this alone wouldn't
    # prove hybrid works) — the real assertion is in the ordering test below.
    assert result[0].source_id == "freiberufler-anmeldung"
    assert {chunk.source_id for chunk in result} <= {"freiberufler-anmeldung", "gbr-gewerbe", "unrelated"}


@pytest.mark.asyncio
async def test_hybrid_retrieve_reorders_relative_to_dense_only_insertion_order() -> None:
    """Seed chunks so the dense-only order (by insertion) disagrees with the correct answer,
    then confirm hybrid + rerank produces the correct order instead."""
    sparse_provider = BM25SparseEmbeddingProvider()
    sync_client = QdrantClient(":memory:")
    ensure_collection(sync_client, COLLECTION, vector_size=4)

    # Insert the unrelated/irrelevant chunk first so a naive dense-only tie-break (insertion
    # order, since all dense vectors are identical) would rank it above the relevant one.
    chunks = [
        _chunk("unrelated", "Der Wetterbericht für Norddeutschland sagt Regen voraus."),
        _chunk(
            "freiberufler-anmeldung",
            "Freiberufler müssen sich beim Finanzamt mit dem Fragebogen zur steuerlichen "
            "Erfassung anmelden.",
        ),
    ]
    dense_vectors = [_UNIFORM_DENSE_VECTOR] * len(chunks)
    sparse_vectors = sparse_provider.embed([chunk.text for chunk in chunks])
    upsert_chunks(sync_client, COLLECTION, chunks, dense_vectors, sparse_vectors)

    retriever = QdrantRetriever(
        SyncQdrantAsyncAdapter(sync_client),  # type: ignore[arg-type]
        COLLECTION,
        stub_dense_embed_fn,
        sparse_provider.embed_query,
        CrossEncoderReranker(),
        candidate_limit=20,
    )

    result = await retriever.retrieve("Wie melde ich mich als Freiberufler beim Finanzamt an?", top_k=1)

    assert result[0].source_id == "freiberufler-anmeldung"
