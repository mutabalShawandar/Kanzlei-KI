import os
from typing import Any

import httpx

from llm.base import ChatMessage, LLMResponse


class OllamaProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.environ["OLLAMA_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["OLLAMA_MODEL"]

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            **kwargs,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            raw=data,
        )
