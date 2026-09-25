import re
from typing import Literal

from pydantic import BaseModel

from llm.base import ChatMessage, LLMProvider
from rag.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are Kanzlei-KI, an assistant for German Freiberufler/GbR registration questions.

Rules, no exceptions:
1. Answer ONLY using the numbered source excerpts provided in the user message. Never use general
   knowledge, training data, or assumptions about German tax/registration law that are not present
   in the excerpts.
2. Every factual claim in your answer must be followed by a bracketed citation marker referencing
   the excerpt it came from, e.g. [1] or [2]. A claim with no marker is treated as unsupported.
3. If the provided excerpts do not answer the question, say exactly:
   "I don't have a reliable source for this." Do not guess, do not fall back to general knowledge.
4. Do not invent excerpt numbers. Only cite numbers that appear in the provided excerpts.
"""

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


class Citation(BaseModel):
    source_id: str
    title: str
    url: str
    section: str | None = None


class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["verified", "unverified"]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered excerpts matching the [n] citation scheme."""
    if not chunks:
        return "No source excerpts were retrieved for this query."

    excerpts = []
    for i, chunk in enumerate(chunks, start=1):
        section_part = f", section {chunk.section}" if chunk.section else ""
        excerpts.append(
            f"[{i}] {chunk.title}{section_part} ({chunk.url})\n{chunk.text}"
        )
    return "\n\n".join(excerpts)


def _extract_citation_numbers(text: str) -> set[int]:
    return {int(n) for n in _CITATION_MARKER_RE.findall(text)}


def _build_validated_answer(raw_answer: str, chunks: list[RetrievedChunk]) -> RAGAnswer:
    cited_numbers = _extract_citation_numbers(raw_answer)
    valid_numbers = {n for n in cited_numbers if 1 <= n <= len(chunks)}

    is_fully_valid = bool(cited_numbers) and cited_numbers == valid_numbers
    confidence: Literal["verified", "unverified"] = (
        "verified" if is_fully_valid else "unverified"
    )

    citations = [
        Citation(
            source_id=chunks[n - 1].source_id,
            title=chunks[n - 1].title,
            url=chunks[n - 1].url,
            section=chunks[n - 1].section,
        )
        for n in sorted(valid_numbers)
    ]

    return RAGAnswer(answer=raw_answer, citations=citations, confidence=confidence)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[ChatMessage]:
    context = format_context(chunks)
    user_content = (
        f"Source excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above, citing every claim with its [n] marker."
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


async def answer_with_citations(
    query: str, chunks: list[RetrievedChunk], llm: LLMProvider
) -> RAGAnswer:
    """Query the LLM with a citation-required prompt and return a validated RAGAnswer.

    `confidence` is derived programmatically from the returned text, never trusted from the
    model's own claim: an answer is only "verified" when every [n] marker it contains maps to
    an actually retrieved chunk, and it contains at least one marker.
    """
    messages = build_messages(query, chunks)
    response = await llm.chat(messages)
    return _build_validated_answer(response.content, chunks)
