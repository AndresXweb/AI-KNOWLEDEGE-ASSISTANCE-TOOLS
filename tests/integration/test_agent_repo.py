
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.agent import get_agent_service
from app.main import app
from app.memory.repository import MemoryRepository
from app.repo_analysis.toolset import RepoToolset
from app.services.agent_service import AgentService


class ScriptedProvider:
    """Primero pide list_files y luego responde."""

    def __init__(self):
        self.calls = []

    async def complete_with_tools(
        self,
        messages,
        tools,
        temperature=0.3,
        top_p=1,
        max_tokens=300,
    ):
        self.calls.append((messages, tools))

        if len(self.calls) == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": "{}",
                        },
                    }
                ],
            }

        return {"content": "El repo tiene un README."}


@pytest.mark.asyncio
async def test_agent_service_runs_repo_tools(tmp_path):
    (tmp_path / "README.md").write_text("hola")
    provider = ScriptedProvider()

    result = await AgentService(provider).ask(
        "¿Qué hay en el repo?",
        extra_tools=RepoToolset(tmp_path),
    )

    messages, tools = provider.calls[1]
    names = {t["function"]["name"] for t in tools}
    tool_message = [m for m in messages if m["role"] == "tool"][0]

    assert result.tools_used == ["list_files"]
    assert {"list_files", "read_file", "search_text",
            "lookup_ai_concept"} <= names
    assert "README.md" in tool_message["content"]
    assert "solo lectura" in messages[0]["content"]


@pytest.mark.asyncio
async def test_agent_without_repo_has_no_repo_tools():
    provider = ScriptedProvider()

    await AgentService(provider).ask("hola")

    _, tools = provider.calls[0]
    names = {t["function"]["name"] for t in tools}

    assert "read_file" not in names


@pytest.mark.asyncio
async def test_agent_endpoint_uses_project_repo(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "mi_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("print('hola')")

    db = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db))

    project = MemoryRepository(db).create_project(
        "Demo", "Proyecto de prueba", str(repo_dir)
    )

    provider = ScriptedProvider()
    app.dependency_overrides[get_agent_service] = (
        lambda: AgentService(provider)
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/agent/chat",
                json={
                    "question": "¿Qué archivos hay?",
                    "project_id": project.id,
                },
            )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    messages, _ = provider.calls[1]
    tool_message = [m for m in messages if m["role"] == "tool"][0]

    assert response.status_code == 200
    assert body["tools_used"] == ["list_files"]
    assert "main.py" in tool_message["content"]
    assert "Proyecto de prueba" in messages[0]["content"]


@pytest.mark.asyncio
async def test_agent_endpoint_unknown_project(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "MEMORY_DB_PATH", str(tmp_path / "memory.db")
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/agent/chat",
            json={"question": "hola", "project_id": 999},
        )

    assert response.status_code == 404
