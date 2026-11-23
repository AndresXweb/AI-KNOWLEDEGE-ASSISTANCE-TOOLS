
import pytest


@pytest.fixture(autouse=True)
def isolated_memory_db(tmp_path, monkeypatch):
    """Evita que las pruebas escriban en data/memory.db real."""
    monkeypatch.setenv(
        "MEMORY_DB_PATH", str(tmp_path / "memory_test.db")
    )
