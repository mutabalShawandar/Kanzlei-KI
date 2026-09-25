from __future__ import annotations

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ingest.chunking import Chunk

_POINT_ID_NAMESPACE = uuid.UUID("d6f6a7d2-8c1e-4b7a-9c3e-8e6a2b1f4d5c")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_EMPTY_SPARSE_VECTOR = SparseVector(indices=[], values=[])


class ChunkVectorCountMismatchError(ValueError):
    pass


class CollectionSchemaMismatchError(ValueError):
    pass


def get_client(url: str | None = None) -> QdrantClient:
    """Build a Qdrant client from an explicit URL or the QDRANT_URL environment variable."""
    return QdrantClient(url=url or os.environ["QDRANT_URL"])


def point_id_for(source_id: str, chunk_index: int) -> str:
    """Derive a stable point ID from a source_id and chunk index so re-ingestion updates points in place."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{source_id}:{chunk_index}"))


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create the hybrid (dense + sparse) collection if it does not already exist.

    Dense remains the primary semantic signal; sparse (BM25-style) is additive, giving exact
    keyword/§-reference precision that dense cosine similarity alone can blur for legal text.
    """
    if client.collection_exists(collection_name):
        _check_hybrid_schema(client, collection_name)
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )


def _check_hybrid_schema(client: QdrantClient, collection_name: str) -> None:
    """Fail loudly if an existing collection predates the hybrid dense+sparse schema.

    `create_collection` is only called once per collection name, so a collection created
    under the old unnamed-dense-vector schema is never migrated automatically; writing named
    vectors into it or reading them back via QdrantRetriever would otherwise fail with an
    unclear Qdrant-side error deep in a request instead of a clear message here.
    """
    info = client.get_collection(collection_name)
    vectors_config = info.config.params.vectors
    sparse_config = info.config.params.sparse_vectors
    has_named_dense = isinstance(vectors_config, dict) and DENSE_VECTOR_NAME in vectors_config
    has_named_sparse = isinstance(sparse_config, dict) and SPARSE_VECTOR_NAME in sparse_config
    if not (has_named_dense and has_named_sparse):
        raise CollectionSchemaMismatchError(
            f"Collection '{collection_name}' predates the hybrid dense+sparse vector schema "
            f"(expected named vectors '{DENSE_VECTOR_NAME}' and '{SPARSE_VECTOR_NAME}'). "
            "Recreate the collection before ingesting or querying it."
        )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    sparse_vectors: list[SparseVector] | None = None,
) -> None:
    """Upsert chunks with their dense (and optional sparse) embeddings, keyed by a deterministic point ID.

    `sparse_vectors` defaults to empty sparse vectors per chunk when omitted, so dense-only
    callers keep working against the hybrid collection schema.
    """
    if len(chunks) != len(vectors):
        raise ChunkVectorCountMismatchError(f"Got {len(chunks)} chunks but {len(vectors)} dense vectors.")
    if sparse_vectors is None:
        sparse_vectors = [_EMPTY_SPARSE_VECTOR] * len(chunks)
    if len(chunks) != len(sparse_vectors):
        raise ChunkVectorCountMismatchError(f"Got {len(chunks)} chunks but {len(sparse_vectors)} sparse vectors.")

    points = [
        PointStruct(
            id=point_id_for(chunk.source_id, chunk.chunk_index),
            vector={DENSE_VECTOR_NAME: vector, SPARSE_VECTOR_NAME: sparse_vector},
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, vector, sparse_vector in zip(chunks, vectors, sparse_vectors, strict=True)
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
