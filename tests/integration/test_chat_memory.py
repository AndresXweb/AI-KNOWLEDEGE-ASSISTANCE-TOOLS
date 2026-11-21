
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.chat import get_assistant_service
from app.api.memory import get_memory_repository
from app.main import app
from app.memory.repository import MemoryRepository
from app.providers.openai_compatible import LLMProviderError
from app.services.assistant_service import AssistantService


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def generate(
        self,
        messages,
        temperature=0.3,
        top_p=1,
        max_tokens=300,
    ):
        self.calls.append(messages)

        if self.fail:
            raise LLMProviderError("caído")

        return f"respuesta {len(self.calls)}"


@pytest.fixture
def env(tmp_path):
    provider = FakeProvider()
    repo = MemoryRepository(tmp_path / "memory.db")

    app.dependency_overrides[get_assistant_service] = (
        lambda: AssistantService(provider)
    )
    app.dependency_overrides[get_memory_repository] = (
        lambda: repo
    )

    yield provider, repo

    app.dependency_overrides.clear()


def client():
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_first_chat_creates_conversation(env):
    provider, repo = env

    async with client() as c:
        r = await c.post(
            "/api/v1/chat", json={"question": "¿Qué es RAG?"}
        )

    body = r.json()

    assert r.status_code == 200
    assert body["answer"] == "respuesta 1"

    saved = repo.get_messages(body["conversation_id"])

    assert [(m.role, m.content) for m in saved] == [
        ("user", "¿Qué es RAG?"),
        ("assistant", "respuesta 1"),
    ]


@pytest.mark.asyncio
async def test_second_chat_sends_previous_history(env):
    provider, repo = env

    async with client() as c:
        first = await c.post(
            "/api/v1/chat", json={"question": "primera"}
        )
        conv_id = first.json()["conversation_id"]

        await c.post(
            "/api/v1/chat",
            json={
                "question": "segunda",
                "conversation_id": conv_id,
            },
        )

    sent = [m["content"] for m in provider.calls[1]]

    assert sent[1:] == ["primera", "respuesta 1", "segunda"]
    assert len(repo.get_messages(conv_id)) == 4


@pytest.mark.asyncio
async def test_unknown_conversation_returns_404(env):
    async with client() as c:
        r = await c.post(
            "/api/v1/chat",
            json={"question": "hola", "conversation_id": 999},
        )

    assert r.status_code == 404


@pytest.mark.asyncio
async def test_unknown_project_returns_404(env):
    async with client() as c:
        r = await c.post(
            "/api/v1/chat",
            json={"question": "hola", "project_id": 999},
        )

    assert r.status_code == 404


@pytest.mark.asyncio
async def test_provider_failure_saves_nothing(env):
    provider, repo = env
    provider.fail = True

    async with client() as c:
        r = await c.post(
            "/api/v1/chat", json={"question": "hola"}
        )

    assert r.status_code == 502

    conversations = repo.list_conversations()

    assert repo.get_messages(conversations[0].id) == []


@pytest.mark.asyncio
async def test_project_context_reaches_the_model(env):
    provider, repo = env

    async with client() as c:
        created = await c.post(
            "/api/v1/projects",
            json={
                "name": "AutoMarketPlus",
                "description": "Marketplace de vehículos",
            },
        )
        project_id = created.json()["id"]

        await c.post(
            "/api/v1/chat",
            json={"question": "hola", "project_id": project_id},
        )

    system = provider.calls[0][0]["content"]

    assert "AutoMarketPlus" in system
    assert "Marketplace de vehículos" in system


@pytest.mark.asyncio
async def test_memory_endpoints(env):
    async with client() as c:
        p = await c.post(
            "/api/v1/projects", json={"name": "P1"}
        )
        dup = await c.post(
            "/api/v1/projects", json={"name": "P1"}
        )
        chat = await c.post(
            "/api/v1/chat",
            json={
                "question": "hola",
                "project_id": p.json()["id"],
            },
        )
        conv_id = chat.json()["conversation_id"]

        convs = await c.get(
            "/api/v1/conversations",
            params={"project_id": p.json()["id"]},
        )
        msgs = await c.get(
            f"/api/v1/conversations/{conv_id}/messages"
        )
        missing = await c.get(
            "/api/v1/conversations/999/messages"
        )
        projects = await c.get("/api/v1/projects")

    assert p.status_code == 201
    assert dup.status_code == 409
    assert [c["id"] for c in convs.json()] == [conv_id]
    assert [m["role"] for m in msgs.json()] == [
        "user", "assistant"
    ]
    assert missing.status_code == 404
    assert [x["name"] for x in projects.json()] == ["P1"]
