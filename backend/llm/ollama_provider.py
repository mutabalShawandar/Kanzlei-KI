import os
from typing import Any

import httpx

from llm.base import ChatMessage, LLMResponse
from llm.network import BaseUrlConfigError, validate_base_url

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OllamaConfigError(BaseUrlConfigError):
    pass


class OllamaUnsupportedRequestError(ValueError):
    pass


class OllamaProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        """Configure the Ollama endpoint and model from arguments or the environment."""
        raw_base_url = (base_url or os.environ["OLLAMA_BASE_URL"]).rstrip("/")
        try:
            self.base_url = validate_base_url(raw_base_url, "OLLAMA_BASE_URL")
        except BaseUrlConfigError as exc:
            raise OllamaConfigError(str(exc)) from exc
        self.model = model or os.environ["OLLAMA_MODEL"]

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Send a non-streaming chat request to Ollama and return its response."""
        if kwargs.pop("stream", False):
            raise OllamaUnsupportedRequestError("Streaming responses are not supported by this provider.")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            **kwargs,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            raw=data,
        )
