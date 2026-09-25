from typing import Any, Literal, Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    raw: dict[str, Any]


class LLMProvider(Protocol):
    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Return a provider response for the supplied chat messages."""
        ...
