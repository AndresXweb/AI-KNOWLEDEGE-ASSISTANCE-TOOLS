
import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from app.api.memory import get_memory_repository
from app.core.config import get_settings
from app.repo_analysis.toolset import RepoToolset

from app.providers.factory import build_provider
from app.providers.openai_compatible import (
    LLMProviderError,
)

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
    service: AgentService = Depends(
        get_agent_service
    ),
) -> AgentChatResponse:

    project = None
    repo_tools = None

    if request.project_id is not None:
        project = get_memory_repository().get_project(
            request.project_id
        )

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Proyecto no encontrado",
            )

        if project.repo_path and Path(project.repo_path).is_dir():
            repo_tools = RepoToolset(project.repo_path)

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

    return AgentChatResponse(
        answer=result.answer,
        model=get_settings().active_model,
        tools_used=result.tools_used,
    )
