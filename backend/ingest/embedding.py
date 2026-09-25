import os
from typing import Protocol

import httpx

from llm.network import BaseUrlConfigError, validate_base_url

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...


class OllamaEmbeddingConfigError(BaseUrlConfigError):
    pass


class OllamaEmbeddingCountMismatchError(ValueError):
    pass


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        """Configure the Ollama embedding endpoint and model from arguments or the environment."""
        raw_base_url = (base_url or os.environ["OLLAMA_BASE_URL"]).rstrip("/")
        try:
            self.base_url = validate_base_url(raw_base_url, "OLLAMA_BASE_URL")
        except BaseUrlConfigError as exc:
            raise OllamaEmbeddingConfigError(str(exc)) from exc
        self.model = model or os.environ["OLLAMA_EMBEDDING_MODEL"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for a batch of texts from Ollama's /api/embed endpoint."""
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(f"{self.base_url}/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()
        embeddings = data["embeddings"]
        if len(embeddings) != len(texts):
            raise OllamaEmbeddingCountMismatchError(
                f"Requested {len(texts)} embeddings but received {len(embeddings)}."
            )
        return embeddings
