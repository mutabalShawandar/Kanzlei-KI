from __future__ import annotations

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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
