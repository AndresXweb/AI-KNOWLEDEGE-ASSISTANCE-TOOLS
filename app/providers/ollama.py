from app.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


class OllamaProvider(OpenAICompatibleProvider):
    """Modelo local servido por Ollama (endpoint compatible
    con OpenAI en /v1). No necesita API key.
    """

    provider_name = "Ollama"
    api_key_env = None

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 120.0,
    ):
        # En CPU los modelos locales tardan más en responder,
        # por eso el timeout por defecto es mayor.
        super().__init__(
            api_key="",
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
