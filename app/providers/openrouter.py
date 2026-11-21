from app.providers.openai_compatible import (
    LLMProviderError,
    OpenAICompatibleProvider,
)

__all__ = ["LLMProviderError", "OpenRouterProvider"]


class OpenRouterProvider(OpenAICompatibleProvider):
    provider_name = "OpenRouter"
    api_key_env = "OPENROUTER_API_KEY"
