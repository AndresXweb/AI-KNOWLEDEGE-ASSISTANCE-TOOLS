
import sqlite3

import pytest

from app.core.config import get_settings
from app.memory.repository import (
    MemoryRepository,
    MemoryStoreError,
)


@pytest.fixture
def repo(tmp_path):
    return MemoryRepository(tmp_path / "data" / "memory.db")


def test_creates_db_file_and_folder(tmp_path):
    path = tmp_path / "nested" / "memory.db"

    MemoryRepository(path)

    assert path.exists()


def test_create_and_get_project(repo):
    project = repo.create_project(
        "AutoMarketPlus", "Marketplace", "C:/repos/auto"
    )

    found = repo.get_project(project.id)

    assert found == project
    assert found.repo_path == "C:/repos/auto"


def test_project_name_is_unique(repo):
    repo.create_project("A")

    with pytest.raises(MemoryStoreError):
        repo.create_project("A")


def test_project_name_required(repo):
    with pytest.raises(MemoryStoreError):
        repo.create_project("   ")


def test_list_projects_sorted_by_name(repo):
    repo.create_project("Zeta")
    repo.create_project("Alfa")

    names = [p.name for p in repo.list_projects()]

    assert names == ["Alfa", "Zeta"]


def test_conversation_with_and_without_project(repo):
    project = repo.create_project("P")

    linked = repo.create_conversation(project.id, "Con proyecto")
    free = repo.create_conversation(title="Suelta")

    assert linked.project_id == project.id
    assert free.project_id is None
    assert len(repo.list_conversations()) == 2
    assert [
        c.id for c in repo.list_conversations(project.id)
    ] == [linked.id]


def test_conversation_requires_existing_project(repo):
    with pytest.raises(MemoryStoreError):
        repo.create_conversation(project_id=999)


def test_messages_keep_chronological_order(repo):
    conv = repo.create_conversation()

    repo.add_message(conv.id, "user", "uno")
    repo.add_message(conv.id, "assistant", "dos", model="m1")
    repo.add_message(conv.id, "user", "tres")

    messages = repo.get_messages(conv.id)

    assert [m.content for m in messages] == [
        "uno", "dos", "tres"
    ]
    assert messages[1].model == "m1"


def test_get_messages_limit_returns_last_n_in_order(repo):
    conv = repo.create_conversation()

    for text in ["a", "b", "c", "d"]:
        repo.add_message(conv.id, "user", text)

    last = repo.get_messages(conv.id, limit=2)

    assert [m.content for m in last] == ["c", "d"]


def test_messages_are_isolated_per_conversation(repo):
    one = repo.create_conversation()
    two = repo.create_conversation()

    repo.add_message(one.id, "user", "solo en uno")

    assert repo.get_messages(two.id) == []


def test_add_message_rejects_invalid_role(repo):
    conv = repo.create_conversation()

    with pytest.raises(MemoryStoreError):
        repo.add_message(conv.id, "robot", "hola")


def test_add_message_requires_existing_conversation(repo):
    with pytest.raises(MemoryStoreError):
        repo.add_message(999, "user", "hola")


def test_add_message_updates_conversation_timestamp(repo):
    conv = repo.create_conversation()

    repo.add_message(conv.id, "user", "hola")

    updated = repo.get_conversation(conv.id)

    assert updated.updated_at >= conv.updated_at


def test_delete_conversation_cascades_to_messages(repo, tmp_path):
    conv = repo.create_conversation()
    repo.add_message(conv.id, "user", "hola")

    assert repo.delete_conversation(conv.id) is True
    assert repo.get_conversation(conv.id) is None
    assert repo.delete_conversation(conv.id) is False

    conn = sqlite3.connect(repo.db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()[0]
    conn.close()

    assert count == 0


def test_data_persists_across_instances(tmp_path):
    path = tmp_path / "memory.db"

    first = MemoryRepository(path)
    conv = first.create_conversation(title="Persistente")
    first.add_message(conv.id, "user", "recuérdame")

    second = MemoryRepository(path)

    assert [
        m.content for m in second.get_messages(conv.id)
    ] == ["recuérdame"]


def test_memory_db_path_setting(monkeypatch):
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    assert get_settings().memory_db_path == "data/memory.db"

    monkeypatch.setenv("MEMORY_DB_PATH", "otro/m.db")
    assert get_settings().memory_db_path == "otro/m.db"
