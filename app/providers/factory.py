from app.core.config import Settings, get_settings
from app.providers.ollama import OllamaProvider
from app.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from app.providers.openrouter import OpenRouterProvider


def build_provider(
    settings: Settings | None = None,
) -> OpenAICompatibleProvider:
    settings = settings or get_settings()

    name = settings.llm_provider.strip().lower()

    if name == "openrouter":
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
        )

    if name == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"LLM_PROVIDER inválido: {settings.llm_provider!r}. "
        "Usa 'openrouter' u 'ollama'."
    )
