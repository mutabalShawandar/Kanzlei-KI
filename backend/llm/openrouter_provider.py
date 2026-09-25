import os
from typing import Any

import httpx

from llm.base import ChatMessage, LLMResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.model = model or os.environ["OPENROUTER_MODEL"]

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers
            )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]
        return LLMResponse(
            content=choice["content"],
            model=data.get("model", self.model),
            raw=data,
        )
