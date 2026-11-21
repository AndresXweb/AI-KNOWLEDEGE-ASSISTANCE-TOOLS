
import json

import httpx
import pytest

from app.core.config import Settings, get_settings
from app.providers.factory import build_provider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compatible import (
    LLMProviderError,
)
from app.providers.openrouter import OpenRouterProvider


def make_settings(**overrides) -> Settings:
    base = dict(
        openrouter_api_key="key",
        openrouter_model="openrouter/free",
        openrouter_base_url="https://example.test/api/v1",
        llm_provider="openrouter",
        ollama_model="llama3.1:8b",
        ollama_base_url="http://localhost:11434/v1",
    )
    base.update(overrides)
    return Settings(**base)


def mock_client(monkeypatch, handler):
    original = httpx.AsyncClient

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )


def ok_response(request):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "hola"}}
            ]
        },
    )


# ---------- configuración ----------

def test_settings_defaults(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.llm_provider == "openrouter"
    assert settings.ollama_base_url.endswith("/v1")


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    settings = get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.active_model == "qwen2.5:7b"


def test_active_model_follows_provider():
    assert (
        make_settings().active_model
        == "openrouter/free"
    )

    assert (
        make_settings(llm_provider="ollama").active_model
        == "llama3.1:8b"
    )


# ---------- fábrica ----------

def test_factory_builds_openrouter():
    provider = build_provider(make_settings())

    assert isinstance(provider, OpenRouterProvider)


def test_factory_builds_ollama():
    provider = build_provider(
        make_settings(llm_provider="Ollama")
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.1:8b"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_provider(
            make_settings(llm_provider="otro")
        )


# ---------- comportamiento HTTP ----------

@pytest.mark.asyncio
async def test_openrouter_requires_api_key():
    provider = OpenRouterProvider(
        "", "openrouter/free", "https://example.test/v1"
    )

    with pytest.raises(LLMProviderError) as exc:
        await provider.generate(
            [{"role": "user", "content": "Hola"}]
        )

    assert "OPENROUTER_API_KEY" in str(exc.value)


@pytest.mark.asyncio
async def test_ollama_works_without_api_key(monkeypatch):
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return ok_response(request)

    mock_client(monkeypatch, handler)

    provider = OllamaProvider(
        "llama3.1:8b", "http://localhost:11434/v1/"
    )

    answer = await provider.generate(
        [{"role": "user", "content": "Hola"}]
    )

    assert answer == "hola"
    assert "authorization" not in seen["headers"]
    assert seen["url"] == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert seen["body"]["model"] == "llama3.1:8b"


@pytest.mark.asyncio
async def test_openrouter_sends_bearer_token(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return ok_response(request)

    mock_client(monkeypatch, handler)

    provider = OpenRouterProvider(
        "secret", "openrouter/free", "https://example.test/v1"
    )

    await provider.generate(
        [{"role": "user", "content": "Hola"}]
    )

    assert seen["auth"] == "Bearer secret"


@pytest.mark.asyncio
async def test_error_message_names_the_provider(monkeypatch):
    mock_client(
        monkeypatch,
        lambda request: httpx.Response(429),
    )

    provider = OllamaProvider(
        "llama3.1:8b", "http://localhost:11434/v1"
    )

    with pytest.raises(LLMProviderError) as exc:
        await provider.generate(
            [{"role": "user", "content": "Hola"}]
        )

    assert "Ollama" in str(exc.value)
    assert "429" in str(exc.value)


@pytest.mark.asyncio
async def test_ollama_supports_tool_calling(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant",
                                 "content": "ok"}}
                ]
            },
        )

    mock_client(monkeypatch, handler)

    provider = OllamaProvider(
        "llama3.1:8b", "http://localhost:11434/v1"
    )

    message = await provider.complete_with_tools(
        messages=[{"role": "user", "content": "Hola"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
    )

    assert message["content"] == "ok"
    assert seen["body"]["tool_choice"] == "auto"
