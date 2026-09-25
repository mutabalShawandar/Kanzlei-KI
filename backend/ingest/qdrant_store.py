from __future__ import annotations

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from ingest.chunking import Chunk

_POINT_ID_NAMESPACE = uuid.UUID("d6f6a7d2-8c1e-4b7a-9c3e-8e6a2b1f4d5c")


class ChunkVectorCountMismatchError(ValueError):
    pass


def get_client(url: str | None = None) -> QdrantClient:
    """Build a Qdrant client from an explicit URL or the QDRANT_URL environment variable."""
    return QdrantClient(url=url or os.environ["QDRANT_URL"])


def point_id_for(source_id: str, chunk_index: int) -> str:
    """Derive a stable point ID from a source_id and chunk index so re-ingestion updates points in place."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{source_id}:{chunk_index}"))


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create the collection if it does not already exist."""
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Upsert chunks and their embeddings as Qdrant points, keyed by a deterministic point ID."""
    if len(chunks) != len(vectors):
        raise ChunkVectorCountMismatchError(f"Got {len(chunks)} chunks but {len(vectors)} vectors.")
    points = [
        PointStruct(
            id=point_id_for(chunk.source_id, chunk.chunk_index),
            vector=vector,
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
    _delete_stale_points(client, collection_name, chunks)


def _delete_stale_points(client: QdrantClient, collection_name: str, chunks: list[Chunk]) -> None:
    """Remove points from a prior ingestion of the same source that the new chunk set no longer produces."""
    indexes_by_source: dict[str, set[int]] = {}
    for chunk in chunks:
        indexes_by_source.setdefault(chunk.source_id, set()).add(chunk.chunk_index)

    for source_id, chunk_indexes in indexes_by_source.items():
        current_point_ids = {point_id_for(source_id, index) for index in chunk_indexes}
        existing_point_ids = _point_ids_for_source(client, collection_name, source_id)
        stale_point_ids = [pid for pid in existing_point_ids if pid not in current_point_ids]
        if stale_point_ids:
            client.delete(collection_name=collection_name, points_selector=stale_point_ids)


def _point_ids_for_source(client: QdrantClient, collection_name: str, source_id: str) -> list[str]:
    point_ids: list[str] = []
    offset = None
    source_filter = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=source_filter,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.extend(str(record.id) for record in records)
        if offset is None:
            break
    return point_ids
