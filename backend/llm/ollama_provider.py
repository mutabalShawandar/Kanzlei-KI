import ipaddress
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from llm.base import ChatMessage, LLMResponse

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OllamaConfigError(ValueError):
    pass


class OllamaUnsupportedRequestError(ValueError):
    pass


def _is_private_network_host(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_private
    except ValueError:
        pass
    # Unqualified single-label hostnames (e.g. Docker Compose service names
    # like "ollama") cannot resolve as public internet FQDNs.
    return "." not in hostname


def _validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme == "https":
        return base_url
    if parsed.scheme == "http" and parsed.hostname is not None and _is_private_network_host(parsed.hostname):
        return base_url
    raise OllamaConfigError(
        f"OLLAMA_BASE_URL must use HTTPS for remote/public hosts (got: {base_url!r}); "
        "plain HTTP is only allowed for loopback, private-network, or unqualified internal hostnames."
    )


class OllamaProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = _validate_base_url((base_url or os.environ["OLLAMA_BASE_URL"]).rstrip("/"))
        self.model = model or os.environ["OLLAMA_MODEL"]

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
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
