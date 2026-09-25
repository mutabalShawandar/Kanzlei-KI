from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel

from ingest.sources.base import SourceDocument

DEFAULT_MAX_CHUNK_CHARS = 1500

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    section: str | None
    retrieved_at: datetime
    chunk_index: int
    text: str


def chunk_document(document: SourceDocument, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a document's text into metadata-preserving chunks along paragraph, then sentence, boundaries."""
    paragraphs = _split_paragraphs(document.text)
    chunk_texts = _pack_paragraphs(paragraphs, max_chunk_chars)
    return [
        Chunk(
            source_id=document.source_id,
            source_type=document.source_type,
            title=document.title,
            url=document.url,
            section=document.section,
            retrieved_at=document.retrieved_at,
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(chunk_texts)
    ]


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _split_sentences(paragraph: str, max_chunk_chars: int) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(paragraph) if sentence.strip()]
    # A single sentence can still exceed max_chunk_chars (no ".", "!", "?" boundary at all);
    # fall back to a hard character split so the configured limit is always enforced.
    return [
        sentence[offset : offset + max_chunk_chars]
        for sentence in sentences
        for offset in range(0, len(sentence), max_chunk_chars)
    ]


def _pack_paragraphs(paragraphs: list[str], max_chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph] if len(paragraph) <= max_chunk_chars else _split_sentences(paragraph, max_chunk_chars)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= max_chunk_chars or not current:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks
