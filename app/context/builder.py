
from app.memory.repository import Message, Project


def _chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)


def build_system_prompt(
    base_prompt: str,
    project: Project | None = None,
) -> str:
    if project is None:
        return base_prompt

    lines = [
        base_prompt,
        "",
        f"Contexto del proyecto «{project.name}»:",
    ]

    if project.description:
        lines.append(project.description)

    if project.repo_path:
        lines.append(f"Ruta del repositorio: {project.repo_path}")

    return "\n".join(lines)


def build_messages(
    system_prompt: str,
    question: str,
    history: list[Message] | None = None,
    project: Project | None = None,
    max_chars: int = 12000,
) -> list[dict[str, str]]:
    """Arma la lista de mensajes para el modelo:
    system (+ proyecto) + historial reciente + pregunta.

    Si el historial no cabe en `max_chars` (aprox. de tokens),
    se descartan primero los mensajes más antiguos.
    """
    system = {
        "role": "system",
        "content": build_system_prompt(system_prompt, project),
    }
    current = {"role": "user", "content": question}

    past = [
        {"role": m.role, "content": m.content}
        for m in (history or [])
        if m.role in ("user", "assistant")
    ]

    budget = max_chars - _chars([system, current])
    kept: list[dict[str, str]] = []

    for message in reversed(past):
        size = len(message["content"])

        if size > budget:
            break

        kept.append(message)
        budget -= size

    kept.reverse()

    return [system, *kept, current]
