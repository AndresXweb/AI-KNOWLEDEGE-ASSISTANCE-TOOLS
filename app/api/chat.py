import logging

from fastapi import APIRouter, Depends, HTTPException

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
from app.schemas.chat import (
    MemoryChatRequest,
    MemoryChatResponse,
)
from app.services.assistant_service import AssistantService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


def get_assistant_service() -> AssistantService:
    # El proveedor se elige con LLM_PROVIDER (openrouter u ollama)
    return AssistantService(build_provider())


@router.post("/chat", response_model=MemoryChatResponse)
async def chat(
    request: MemoryChatRequest,
    service: AssistantService = Depends(get_assistant_service),
    repo: MemoryRepository = Depends(get_memory_repository),
):
    settings = get_settings()

    conversation, project = resolve_conversation(
        repo,
        request.conversation_id,
        request.project_id,
        request.question,
    )

    # El historial se lee ANTES de guardar la pregunta nueva.
    history = repo.get_messages(
        conversation.id,
        limit=settings.history_max_messages,
    )

    try:
        answer = await service.ask(
            question=request.question,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            history=history,
            project=project,
            max_chars=settings.context_max_chars,
        )

    except LLMProviderError as exc:
        logger.error(
            "Fallo del proveedor LLM: %s",
            exc,
            exc_info=exc.__cause__,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "No fue posible obtener una respuesta "
                f"del LLM: {exc}"
            ),
        )

    # Solo se guarda el intercambio si el modelo respondió.
    model = settings.active_model

    repo.add_message(conversation.id, "user", request.question)
    repo.add_message(
        conversation.id, "assistant", answer, model=model
    )

    return MemoryChatResponse(
        answer=answer,
        model=model,
        conversation_id=conversation.id,
    )
