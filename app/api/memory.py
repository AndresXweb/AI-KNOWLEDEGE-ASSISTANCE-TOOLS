from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.memory.repository import (
    Conversation,
    MemoryRepository,
    MemoryStoreError,
    Project,
)
from app.schemas.memory import (
    ConversationOut,
    MessageOut,
    ProjectCreate,
    ProjectOut,
)


router = APIRouter(prefix="/api/v1", tags=["Memoria"])


def get_memory_repository() -> MemoryRepository:
    return MemoryRepository(get_settings().memory_db_path)


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    repo: MemoryRepository = Depends(get_memory_repository),
):
    try:
        project = repo.create_project(
            body.name, body.description, body.repo_path
        )
    except MemoryStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return project.__dict__


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(
    repo: MemoryRepository = Depends(get_memory_repository),
):
    return [p.__dict__ for p in repo.list_projects()]


@router.get(
    "/conversations", response_model=list[ConversationOut]
)
async def list_conversations(
    project_id: int | None = None,
    repo: MemoryRepository = Depends(get_memory_repository),
):
    return [
        c.__dict__ for c in repo.list_conversations(project_id)
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
async def conversation_messages(
    conversation_id: int,
    repo: MemoryRepository = Depends(get_memory_repository),
):
    if repo.get_conversation(conversation_id) is None:
        raise HTTPException(
            status_code=404, detail="Conversación no encontrada"
        )

    return [
        m.__dict__ for m in repo.get_messages(conversation_id)
    ]


def resolve_conversation(
    repo: MemoryRepository,
    conversation_id: int | None,
    project_id: int | None,
    title: str,
) -> tuple[Conversation, Project | None]:
    """Devuelve la conversación (existente o nueva) y su proyecto.

    Con `conversation_id` se ignora `project_id`: la conversación
    conserva el proyecto con el que fue creada.
    """
    if conversation_id is not None:
        conversation = repo.get_conversation(conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversación no encontrada",
            )
    else:
        try:
            conversation = repo.create_conversation(
                project_id=project_id,
                title=title[:60],
            )
        except MemoryStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    project = None

    if conversation.project_id is not None:
        project = repo.get_project(conversation.project_id)

    return conversation, project
