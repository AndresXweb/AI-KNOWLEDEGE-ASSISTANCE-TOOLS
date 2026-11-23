
import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.agent import get_agent_service
from app.main import app
from app.memory.repository import MemoryRepository
from app.providers.openai_compatible import LLMProviderError
from app.services.agent_service import AgentService


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def complete_with_tools(
        self,
        messages,
        tools,
        temperature=0.3,
        top_p=1,
        max_tokens=300,
    ):
        self.calls.append((messages, tools))

        if self.fail:
            raise LLMProviderError("caído")

        return {"content": f"respuesta {len(self.calls)}"}


@pytest.fixture
def provider():
    fake = FakeProvider()

    app.dependency_overrides[get_agent_service] = (
        lambda: AgentService(fake)
    )

    yield fake

    app.dependency_overrides.clear()


def memory() -> MemoryRepository:
    return MemoryRepository(os.environ["MEMORY_DB_PATH"])


def client():
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_agent_saves_exchange_and_returns_conversation(
    provider,
):
    async with client() as c:
        r = await c.post(
            "/api/v1/agent/chat", json={"question": "hola"}
        )

    body = r.json()
    saved = memory().get_messages(body["conversation_id"])

    assert r.status_code == 200
    assert [(m.role, m.content) for m in saved] == [
        ("user", "hola"),
        ("assistant", "respuesta 1"),
    ]


@pytest.mark.asyncio
async def test_agent_receives_previous_history(provider):
    async with client() as c:
        first = await c.post(
            "/api/v1/agent/chat", json={"question": "primera"}
        )
        conv_id = first.json()["conversation_id"]

        await c.post(
            "/api/v1/agent/chat",
            json={
                "question": "segunda",
                "conversation_id": conv_id,
            },
        )

    sent = [m["content"] for m in provider.calls[1][0]]

    assert sent[1:] == ["primera", "respuesta 1", "segunda"]


@pytest.mark.asyncio
async def test_conversation_keeps_its_project_repo_tools(
    provider, tmp_path
):
    repo_dir = tmp_path / "mi_repo"
    repo_dir.mkdir()

    project = memory().create_project(
        "Demo", "Desc", str(repo_dir)
    )

    async with client() as c:
        first = await c.post(
            "/api/v1/agent/chat",
            json={"question": "uno", "project_id": project.id},
        )

        # Segunda pregunta SIN project_id: hereda el de la conversación.
        await c.post(
            "/api/v1/agent/chat",
            json={
                "question": "dos",
                "conversation_id": first.json()["conversation_id"],
            },
        )

    for messages, tools in provider.calls:
        names = {t["function"]["name"] for t in tools}

        assert "read_file" in names
        assert "Desc" in messages[0]["content"]


@pytest.mark.asyncio
async def test_unknown_conversation_returns_404(provider):
    async with client() as c:
        r = await c.post(
            "/api/v1/agent/chat",
            json={"question": "hola", "conversation_id": 999},
        )

    assert r.status_code == 404


@pytest.mark.asyncio
async def test_agent_failure_saves_nothing(provider):
    provider.fail = True

    async with client() as c:
        r = await c.post(
            "/api/v1/agent/chat", json={"question": "hola"}
        )

    conversations = memory().list_conversations()

    assert r.status_code == 502
    assert memory().get_messages(conversations[0].id) == []


@pytest.mark.asyncio
async def test_chat_and_agent_share_conversation_history(provider):
    from app.api.chat import get_assistant_service
    from app.services.assistant_service import AssistantService

    class ChatFake:
        def __init__(self):
            self.calls = []

        async def generate(
            self, messages, temperature=0.3, top_p=1,
            max_tokens=300,
        ):
            self.calls.append(messages)
            return "desde chat"

    chat_fake = ChatFake()
    app.dependency_overrides[get_assistant_service] = (
        lambda: AssistantService(chat_fake)
    )

    async with client() as c:
        first = await c.post(
            "/api/v1/agent/chat", json={"question": "del agente"}
        )
        await c.post(
            "/api/v1/chat",
            json={
                "question": "del chat",
                "conversation_id": first.json()["conversation_id"],
            },
        )

    sent = [m["content"] for m in chat_fake.calls[0]]

    assert sent[1:] == [
        "del agente", "respuesta 1", "del chat"
    ]
