import json

import httpx
import pytest
import respx

from llm.base import ChatMessage
from llm.openrouter_provider import (
    OpenRouterProvider,
    OpenRouterResponseError,
    OpenRouterUnsupportedRequestError,
)


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_expected_request_and_parses_response() -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "openai/gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Hallo"}}],
            },
        )
    )
    provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4o-mini")

    result = await provider.chat([ChatMessage(role="user", content="Hi")])

    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(httpx.Request.content.fget(request))
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]

    assert result.content == "Hallo"
    assert result.model == "openai/gpt-4o-mini"
    assert result.raw["choices"][0]["message"]["content"] == "Hallo"


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_on_http_error() -> None:
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401)
    )
    provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4o-mini")

    with pytest.raises(httpx.HTTPStatusError):
        await provider.chat([ChatMessage(role="user", content="Hi")])


def test_reads_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")

    provider = OpenRouterProvider()

    assert provider.api_key == "env-key"
    assert provider.model == "env-model"


@pytest.mark.asyncio
async def test_rejects_streaming_request() -> None:
    provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4o-mini")

    with pytest.raises(OpenRouterUnsupportedRequestError):
        await provider.chat([ChatMessage(role="user", content="Hi")], stream=True)


@pytest.mark.asyncio
async def test_rejects_tool_calling_request() -> None:
    provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4o-mini")

    with pytest.raises(OpenRouterUnsupportedRequestError):
        await provider.chat([ChatMessage(role="user", content="Hi")], tools=[{"type": "function"}])


@pytest.mark.asyncio
@respx.mock
async def test_raises_on_null_content_response() -> None:
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "openai/gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": None}}],
            },
        )
    )
    provider = OpenRouterProvider(api_key="test-key", model="openai/gpt-4o-mini")

    with pytest.raises(OpenRouterResponseError):
        await provider.chat([ChatMessage(role="user", content="Hi")])
