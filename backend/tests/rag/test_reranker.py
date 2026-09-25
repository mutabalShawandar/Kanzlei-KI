import pytest

from rag.reranker import CrossEncoderReranker
from rag.retrieval import RetrievedChunk


def make_chunk(text: str, source_id: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source_id=source_id, title=source_id, url=f"https://example.com/{source_id}")


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.mark.asyncio
async def test_rerank_returns_empty_list_for_no_candidates(reranker: CrossEncoderReranker) -> None:
    result = await reranker.rerank("query", [], top_k=5)

    assert result == []


@pytest.mark.asyncio
async def test_rerank_rejects_non_positive_top_k(reranker: CrossEncoderReranker) -> None:
    with pytest.raises(ValueError):
        await reranker.rerank("query", [make_chunk("text", "a")], top_k=0)


@pytest.mark.asyncio
async def test_rerank_truncates_to_top_k(reranker: CrossEncoderReranker) -> None:
    candidates = [make_chunk(f"chunk {i}", f"c{i}") for i in range(5)]

    result = await reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2


@pytest.mark.asyncio
async def test_rerank_ranks_relevant_german_passage_above_unrelated_one(reranker: CrossEncoderReranker) -> None:
    relevant = make_chunk(
        "Freiberufler müssen ihre Tätigkeit beim zuständigen Finanzamt anmelden, "
        "indem sie den Fragebogen zur steuerlichen Erfassung ausfüllen.",
        source_id="relevant",
    )
    unrelated = make_chunk(
        "Das Bundesministerium für Ernährung veröffentlicht jährlich Statistiken "
        "zum Konsum von Milchprodukten in Deutschland.",
        source_id="unrelated",
    )

    result = await reranker.rerank(
        "Wie melde ich mich als Freiberufler beim Finanzamt an?",
        [unrelated, relevant],
        top_k=2,
    )

    assert result[0].source_id == "relevant"
