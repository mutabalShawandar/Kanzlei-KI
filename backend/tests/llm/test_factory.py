import pytest

from llm.factory import get_llm_provider
from llm.ollama_provider import OllamaProvider
from llm.openrouter_provider import OpenRouterProvider


def test_returns_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")

    provider = get_llm_provider()

    assert isinstance(provider, OllamaProvider)


def test_returns_openrouter_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    provider = get_llm_provider()

    assert isinstance(provider, OpenRouterProvider)


def test_raises_on_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bogus")

    with pytest.raises(ValueError, match="Unknown or missing LLM_PROVIDER"):
        get_llm_provider()


def test_raises_on_missing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    with pytest.raises(ValueError, match="Unknown or missing LLM_PROVIDER"):
        get_llm_provider()
