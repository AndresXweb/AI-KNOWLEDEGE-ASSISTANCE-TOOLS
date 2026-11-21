from typing import Any

import httpx


class LLMProviderError(Exception):
    pass


class OpenAICompatibleProvider:
    """Cliente base para cualquier API compatible con
    /chat/completions de OpenAI (OpenRouter, Ollama, etc.).
    """

    provider_name = "LLM"

    # Nombre de la variable de entorno de la API key.
    # None significa que el proveedor no exige llave (Ollama).
    api_key_env: str | None = None

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _request(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        if self.api_key_env and not self.api_key:
            raise LLMProviderError(
                f"No se configuró {self.api_key_env}"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                "Tiempo de espera agotado"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"{self.provider_name} respondió HTTP "
                f"{exc.response.status_code}"
            ) from exc

        except (
            httpx.RequestError,
            ValueError,
        ) as exc:
            raise LLMProviderError(
                "Error de comunicación con "
                f"{self.provider_name}"
            ) from exc

        try:
            message = data["choices"][0]["message"]

            if not isinstance(message, dict):
                raise TypeError("Respuesta inválida")

            return message

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise LLMProviderError(
                "Formato de respuesta LLM inválido"
            ) from exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> str:

        message = await self._request(
            messages,
            temperature,
            top_p,
            max_tokens,
        )

        content = message.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise LLMProviderError(
                "Respuesta de texto vacía"
            )

        return content

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> dict[str, Any]:

        return await self._request(
            messages,
            temperature,
            top_p,
            max_tokens,
            tools=tools,
        )
