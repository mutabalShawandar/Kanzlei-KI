from typing import Any

import pytest

from llm.base import ChatMessage, LLMResponse
from rag.prompting import RAGAnswer, answer_with_citations, format_context
from rag.retrieval import RetrievedChunk


class StubLLMProvider:
    """Fake LLMProvider returning a canned response text, no real LLM call."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.received_messages: list[ChatMessage] = []

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        self.received_messages = messages
        return LLMResponse(content=self._response_text, model="stub", raw={})


def make_chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"Excerpt text {n}",
        source_id=f"source-{n}",
        title=f"Title {n}",
        url=f"https://example.com/{n}",
        section=f"§{n} EStG",
    )


@pytest.mark.asyncio
async def test_well_formed_answer_with_valid_citations_is_verified() -> None:
    chunks = [make_chunk(1), make_chunk(2)]
    llm = StubLLMProvider("Freiberufler must register with the Finanzamt [1]. No trade license is needed [2].")

    result = await answer_with_citations("Do I need to register?", chunks, llm)

    assert isinstance(result, RAGAnswer)
    assert result.confidence == "verified"
    assert len(result.citations) == 2
    assert result.citations[0].source_id == "source-1"
    assert result.citations[1].source_id == "source-2"
    assert result.citations[0].section == "§1 EStG"


@pytest.mark.asyncio
async def test_answer_with_no_citation_markers_is_unverified() -> None:
    chunks = [make_chunk(1)]
    llm = StubLLMProvider("Freiberufler must register with the Finanzamt.")

    result = await answer_with_citations("Do I need to register?", chunks, llm)

    assert result.confidence == "unverified"
    assert result.citations == []


@pytest.mark.asyncio
async def test_answer_citing_nonexistent_marker_is_unverified() -> None:
    chunks = [make_chunk(1), make_chunk(2)]
    llm = StubLLMProvider("This claim cites a source that was never given [5].")

    result = await answer_with_citations("query", chunks, llm)

    assert result.confidence == "unverified"
    assert result.citations == []


@pytest.mark.asyncio
async def test_answer_mixing_valid_and_invalid_markers_is_unverified() -> None:
    chunks = [make_chunk(1), make_chunk(2)]
    llm = StubLLMProvider("Partly supported [1], partly fabricated [9].")

    result = await answer_with_citations("query", chunks, llm)

    assert result.confidence == "unverified"
    assert [c.source_id for c in result.citations] == ["source-1"]


@pytest.mark.asyncio
async def test_empty_chunks_never_crashes_and_is_unverified() -> None:
    llm = StubLLMProvider("I don't have a reliable source for this.")

    result = await answer_with_citations("query", [], llm)

    assert isinstance(result, RAGAnswer)
    assert result.confidence == "unverified"
    assert result.citations == []


@pytest.mark.asyncio
async def test_empty_chunks_with_fabricated_citation_marker_is_unverified() -> None:
    llm = StubLLMProvider("Some fabricated fact [1].")

    result = await answer_with_citations("query", [], llm)

    assert result.confidence == "unverified"
    assert result.citations == []


def test_format_context_numbers_excerpts_matching_citation_scheme() -> None:
    chunks = [make_chunk(1), make_chunk(2)]

    context = format_context(chunks)

    assert "[1] Title 1, section §1 EStG (https://example.com/1)" in context
    assert "[2] Title 2, section §2 EStG (https://example.com/2)" in context


def test_format_context_handles_no_chunks() -> None:
    context = format_context([])

    assert "No source excerpts" in context


@pytest.mark.asyncio
async def test_prompt_instructs_citation_required_behavior() -> None:
    llm = StubLLMProvider("Answer [1].")
    chunks = [make_chunk(1)]

    await answer_with_citations("query", chunks, llm)

    system_message = llm.received_messages[0]
    assert system_message.role == "system"
    assert "cite" in system_message.content.lower()
    assert "I don't have a reliable source for this." in system_message.content
