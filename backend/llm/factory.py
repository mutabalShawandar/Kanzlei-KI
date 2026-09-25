import os

from llm.base import LLMProvider
from llm.ollama_provider import OllamaProvider
from llm.openrouter_provider import OpenRouterProvider


def get_llm_provider() -> LLMProvider:
    """Create the provider selected by the LLM_PROVIDER environment variable."""
    provider_name = os.environ.get("LLM_PROVIDER")
    if provider_name == "ollama":
        return OllamaProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()
    raise ValueError(
        f"Unknown or missing LLM_PROVIDER: {provider_name!r}. Expected 'ollama' or 'openrouter'."
    )
