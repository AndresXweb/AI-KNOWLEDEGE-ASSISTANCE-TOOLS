
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.memory.db import connect, init_db


VALID_ROLES = {"system", "user", "assistant", "tool"}


class MemoryStoreError(Exception):
    pass


@dataclass(frozen=True)
class Project:
    id: int
    name: str
    description: str
    repo_path: str | None
    created_at: str


@dataclass(frozen=True)
class Conversation:
    id: int
    project_id: int | None
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    model: str | None
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryRepository:
    """Almacena proyectos, conversaciones y mensajes
    en un archivo SQLite local.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    # ---------- proyectos ----------

    def create_project(
        self,
        name: str,
        description: str = "",
        repo_path: str | None = None,
    ) -> Project:
        name = name.strip()

        if not name:
            raise MemoryStoreError(
                "El nombre del proyecto es obligatorio"
            )

        try:
            with connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO projects "
                    "(name, description, repo_path, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name, description, repo_path, _now()),
                )
                row = conn.execute(
                    "SELECT * FROM projects WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(
                f"Ya existe un proyecto llamado {name!r}"
            ) from exc

        return Project(**row)

    def get_project(self, project_id: int) -> Project | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()

        return Project(**row) if row else None

    def list_projects(self) -> list[Project]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY name"
            ).fetchall()

        return [Project(**row) for row in rows]

    # ---------- conversaciones ----------

    def create_conversation(
        self,
        project_id: int | None = None,
        title: str = "Nueva conversación",
    ) -> Conversation:
        now = _now()

        try:
            with connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO conversations "
                    "(project_id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (project_id, title.strip() or "Sin título",
                     now, now),
                )
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(
                f"El proyecto {project_id} no existe"
            ) from exc

        return Conversation(**row)

    def get_conversation(
        self, conversation_id: int
    ) -> Conversation | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

        return Conversation(**row) if row else None

    def list_conversations(
        self, project_id: int | None = None
    ) -> list[Conversation]:
        query = "SELECT * FROM conversations"
        params: tuple = ()

        if project_id is not None:
            query += " WHERE project_id = ?"
            params = (project_id,)

        query += " ORDER BY updated_at DESC, id DESC"

        with connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [Conversation(**row) for row in rows]

    def delete_conversation(self, conversation_id: int) -> bool:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )

        return cursor.rowcount > 0

    # ---------- mensajes ----------

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model: str | None = None,
    ) -> Message:
        if role not in VALID_ROLES:
            raise MemoryStoreError(
                f"Rol inválido: {role!r}"
            )

        now = _now()

        try:
            with connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO messages "
                    "(conversation_id, role, content, model, "
                    "created_at) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, role, content, model, now),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? "
                    "WHERE id = ?",
                    (now, conversation_id),
                )
                row = conn.execute(
                    "SELECT * FROM messages WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(
                f"La conversación {conversation_id} no existe"
            ) from exc

        return Message(**row)

    def get_messages(
        self,
        conversation_id: int,
        limit: int | None = None,
    ) -> list[Message]:
        """Devuelve los mensajes en orden cronológico.
        Con `limit`, devuelve solo los últimos N.
        """
        with connect(self.db_path) as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT * FROM messages "
                    "WHERE conversation_id = ? ORDER BY id",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ("
                    "SELECT * FROM messages "
                    "WHERE conversation_id = ? "
                    "ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id",
                    (conversation_id, limit),
                ).fetchall()

        return [Message(**row) for row in rows]
