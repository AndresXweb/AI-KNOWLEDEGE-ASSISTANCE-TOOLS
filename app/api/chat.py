
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.providers.factory import build_provider
from app.providers.openai_compatible import (
    LLMProviderError,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.assistant_service import AssistantService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


def get_assistant_service() -> AssistantService: 
    # El proveedor se elige con LLM_PROVIDER (openrouter u ollama)
    return AssistantService(build_provider())


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: AssistantService = Depends(
        get_assistant_service
    ),
):
    try:
        answer = await service.ask(
            question=request.question,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

    except LLMProviderError as exc:
        logger.error("Fallo del proveedor LLM: %s", exc, exc_info=exc.__cause__)
        raise HTTPException(
            status_code=502,
            detail=f"No fue posible obtener una respuesta del LLM: {exc}",
        )

    return ChatResponse(
        answer=answer,
        model=get_settings().active_model,
    )
