from datetime import datetime, timezone

from ingest.chunking import chunk_document
from ingest.sources.base import SourceDocument

RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _document(text: str, **overrides: object) -> SourceDocument:
    fields = {
        "source_id": "estg-1",
        "source_type": "statute",
        "title": "EStG §1",
        "url": "https://gesetze-im-internet.de/estg/__1.html",
        "section": "§1",
        "text": text,
        "retrieved_at": RETRIEVED_AT,
    }
    fields.update(overrides)
    return SourceDocument(**fields)


def test_splits_on_paragraph_boundaries_when_within_limit() -> None:
    document = _document("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")

    chunks = chunk_document(document, max_chunk_chars=1000)

    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."


def test_packs_paragraphs_until_limit_then_starts_new_chunk() -> None:
    document = _document("Para one.\n\nPara two.\n\nPara three.")

    chunks = chunk_document(document, max_chunk_chars=9)

    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["Para one.", "Para two.", "Para three."]


def test_does_not_split_mid_sentence_within_an_oversized_paragraph() -> None:
    long_paragraph = "Sentence one is here. Sentence two is here too. Sentence three follows."
    document = _document(long_paragraph)

    chunks = chunk_document(document, max_chunk_chars=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.strip().endswith((".", "!", "?"))


def test_chunk_index_is_sequential() -> None:
    document = _document("Para one.\n\nPara two.\n\nPara three.")

    chunks = chunk_document(document, max_chunk_chars=15)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_carries_forward_parent_metadata() -> None:
    document = _document("Only paragraph.", section="§1 Abs. 1")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_id == document.source_id
    assert chunk.source_type == document.source_type
    assert chunk.title == document.title
    assert chunk.url == document.url
    assert chunk.section == "§1 Abs. 1"
    assert chunk.retrieved_at == document.retrieved_at


def test_empty_text_produces_no_chunks() -> None:
    document = _document("   \n\n  ")

    chunks = chunk_document(document)

    assert chunks == []
