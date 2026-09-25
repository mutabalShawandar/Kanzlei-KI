import os
from typing import Any

import httpx

from llm.base import ChatMessage, LLMResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenRouterUnsupportedRequestError(ValueError):
    pass


class OpenRouterResponseError(RuntimeError):
    pass


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Configure the OpenRouter API key and model from arguments or the environment."""
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.model = model or os.environ["OPENROUTER_MODEL"]

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Send a chat completion request and return its non-tool response."""
        if kwargs.get("stream"):
            raise OpenRouterUnsupportedRequestError("Streaming responses are not supported by this provider.")
        if "tools" in kwargs:
            raise OpenRouterUnsupportedRequestError("Tool-calling requests are not supported by this provider.")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers
            )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]
        content = choice["content"]
        if content is None:
            raise OpenRouterResponseError(
                "OpenRouter returned a null-content message (likely a tool call), which this provider cannot represent."
            )
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            raw=data,
        )
