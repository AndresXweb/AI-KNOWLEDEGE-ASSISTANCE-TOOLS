import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from app.api.memory import (
    get_memory_repository,
    resolve_conversation,
)
from app.core.config import get_settings
from app.memory.repository import MemoryRepository
from app.providers.factory import build_provider
from app.providers.openai_compatible import (
    LLMProviderError,
)
from app.repo_analysis.toolset import RepoToolset
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
)
from app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/agent",
    tags=["Agent (Módulo III)"],
)


def get_agent_service() -> AgentService:
    # El proveedor se elige con LLM_PROVIDER (openrouter u ollama)
    return AgentService(build_provider())


@router.post(
    "/chat",
    response_model=AgentChatResponse,
)
async def chat_agent(
    request: AgentChatRequest,
    service: AgentService = Depends(get_agent_service),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> AgentChatResponse:

    settings = get_settings()

    conversation, project = resolve_conversation(
        repo,
        request.conversation_id,
        request.project_id,
        request.question,
    )

    repo_tools = None

    if (
        project is not None
        and project.repo_path
        and Path(project.repo_path).is_dir()
    ):
        repo_tools = RepoToolset(project.repo_path)

    # El historial se lee ANTES de guardar la pregunta nueva.
    history = repo.get_messages(
        conversation.id,
        limit=settings.history_max_messages,
    )

    try:
        result = await service.ask(
            question=request.question,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            confirm_actions=request.confirm_actions,
            max_steps=request.max_steps,
            project=project,
            extra_tools=repo_tools,
            history=history,
            max_chars=settings.context_max_chars,
        )

    except (
        AgentExecutionError,
        LLMProviderError,
    ) as exc:

        logger.warning(
            "Fallo del agente: %s",
            exc,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "No se pudo completar "
                "la ejecución del agente"
            ),
        ) from exc

    # Solo se guarda el intercambio si el agente respondió.
    model = settings.active_model

    repo.add_message(
        conversation.id, "user", request.question
    )
    repo.add_message(
        conversation.id,
        "assistant",
        result.answer,
        model=model,
    )

    return AgentChatResponse(
        answer=result.answer,
        model=model,
        tools_used=result.tools_used,
        conversation_id=conversation.id,
    )
