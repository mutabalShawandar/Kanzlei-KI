from __future__ import annotations

from qdrant_client import QdrantClient

from ingest import qdrant_store
from ingest.chunking import chunk_document
from ingest.embedding import EmbeddingProvider
from ingest.sources.base import SourceDocument

_EMBEDDING_BATCH_SIZE = 64


class EmptyIngestionBatchError(ValueError):
    pass


async def _embed_in_batches(embedding_provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
        vectors.extend(await embedding_provider.embed(batch))
    return vectors


async def ingest_documents(
    documents: list[SourceDocument],
    collection_name: str,
    embedding_provider: EmbeddingProvider,
    client: QdrantClient | None = None,
) -> int:
    """Chunk, embed, and upsert a batch of source documents; return the number of chunks upserted."""
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    if not chunks:
        raise EmptyIngestionBatchError("No chunks were produced from the supplied documents.")

    vectors = await _embed_in_batches(embedding_provider, [chunk.text for chunk in chunks])

    qdrant_client = client or qdrant_store.get_client()
    qdrant_store.ensure_collection(qdrant_client, collection_name, vector_size=len(vectors[0]))
    qdrant_store.upsert_chunks(qdrant_client, collection_name, chunks, vectors)
    return len(chunks)
