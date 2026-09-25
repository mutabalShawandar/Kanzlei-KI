import httpx
import pytest
import respx

from llm.base import ChatMessage
from llm.ollama_provider import (
    OllamaConfigError,
    OllamaProvider,
    OllamaUnsupportedRequestError,
)


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_expected_request_and_parses_response() -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3",
                "message": {"role": "assistant", "content": "Hallo"},
            },
        )
    )
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    result = await provider.chat([ChatMessage(role="user", content="Hi")])

    assert route.called
    request = route.calls.last.request
    assert request.url == "http://localhost:11434/api/chat"
    body = httpx.Request.content.fget(request)
    import json

    payload = json.loads(body)
    assert payload["model"] == "llama3"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert payload["stream"] is False

    assert result.content == "Hallo"
    assert result.model == "llama3"
    assert result.raw["message"]["content"] == "Hallo"


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_on_http_error() -> None:
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(500))
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    with pytest.raises(httpx.HTTPStatusError):
        await provider.chat([ChatMessage(role="user", content="Hi")])


def test_reads_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://envhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")

    provider = OllamaProvider()

    assert provider.base_url == "https://envhost:11434"
    assert provider.model == "env-model"


def test_rejects_remote_http_base_url() -> None:
    with pytest.raises(OllamaConfigError):
        OllamaProvider(base_url="http://public.example.com:11434", model="llama3")


def test_allows_http_for_docker_compose_service_name() -> None:
    provider = OllamaProvider(base_url="http://ollama:11434", model="llama3")
    assert provider.base_url == "http://ollama:11434"


def test_allows_http_for_private_network_ip() -> None:
    provider = OllamaProvider(base_url="http://192.168.1.5:11434", model="llama3")
    assert provider.base_url == "http://192.168.1.5:11434"


def test_rejects_http_for_public_ip() -> None:
    with pytest.raises(OllamaConfigError):
        OllamaProvider(base_url="http://8.8.8.8:11434", model="llama3")


def test_allows_https_remote_base_url() -> None:
    provider = OllamaProvider(base_url="https://envhost:11434", model="llama3")
    assert provider.base_url == "https://envhost:11434"


@pytest.mark.asyncio
async def test_rejects_streaming_request() -> None:
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    with pytest.raises(OllamaUnsupportedRequestError):
        await provider.chat([ChatMessage(role="user", content="Hi")], stream=True)
