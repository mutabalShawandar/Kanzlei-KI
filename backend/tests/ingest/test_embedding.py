import httpx
import pytest
import respx

from ingest.embedding import (
    OllamaEmbeddingConfigError,
    OllamaEmbeddingCountMismatchError,
    OllamaEmbeddingProvider,
)


@pytest.mark.asyncio
@respx.mock
async def test_embed_sends_expected_request_and_parses_response() -> None:
    """Send Ollama's expected embed payload and parse its response."""
    route = respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

    result = await provider.embed(["first", "second"])

    assert route.called
    request = route.calls.last.request
    import json

    payload = json.loads(httpx.Request.content.fget(request))
    assert payload == {"model": "nomic-embed-text", "input": ["first", "second"]}
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
@respx.mock
async def test_embed_raises_on_http_error() -> None:
    """Propagate Ollama HTTP errors to the caller."""
    respx.post("http://localhost:11434/api/embed").mock(return_value=httpx.Response(500))
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed(["hi"])


@pytest.mark.asyncio
@respx.mock
async def test_embed_raises_on_count_mismatch() -> None:
    """Fail loudly if Ollama returns a different number of embeddings than requested."""
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

    with pytest.raises(OllamaEmbeddingCountMismatchError):
        await provider.embed(["first", "second"])


def test_reads_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the Ollama embedding URL and model from the environment."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://envhost:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "env-embed-model")

    provider = OllamaEmbeddingProvider()

    assert provider.base_url == "https://envhost:11434"
    assert provider.model == "env-embed-model"


def test_rejects_remote_http_base_url() -> None:
    """Reject plain HTTP for a public Ollama hostname."""
    with pytest.raises(OllamaEmbeddingConfigError):
        OllamaEmbeddingProvider(base_url="http://public.example.com:11434", model="nomic-embed-text")


def test_allows_http_for_docker_compose_service_name() -> None:
    """Allow plain HTTP for an internal single-label hostname."""
    provider = OllamaEmbeddingProvider(base_url="http://ollama:11434", model="nomic-embed-text")
    assert provider.base_url == "http://ollama:11434"
