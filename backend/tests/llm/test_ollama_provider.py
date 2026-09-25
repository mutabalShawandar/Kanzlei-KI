import httpx
import pytest
import respx

from llm.base import ChatMessage
from llm.ollama_provider import OllamaProvider


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
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://envhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")

    provider = OllamaProvider()

    assert provider.base_url == "http://envhost:11434"
    assert provider.model == "env-model"
