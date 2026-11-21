from app.context.builder import build_messages
from app.memory.repository import Message, Project
from app.providers.base import LLMProvider


SYSTEM_PROMPT = """
Eres AI Knowledge Assistant, un asistente
educativo especializado en tecnología.

Responde en español claro.
Explica los conceptos con ejemplos sencillos.
Si no tienes suficiente información, indícalo.
No inventes referencias ni datos.
""".strip()


class AssistantService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def ask(
        self,
        question: str,
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
        history: list[Message] | None = None,
        project: Project | None = None,
        max_chars: int = 12000,
    ) -> str:

        messages = build_messages(
            SYSTEM_PROMPT,
            question,
            history=history,
            project=project,
            max_chars=max_chars,
        )

        return await self.provider.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
