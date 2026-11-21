
from app.context.builder import (
    build_messages,
    build_system_prompt,
)
from app.memory.repository import Message, Project


def msg(i, role, content):
    return Message(
        id=i,
        conversation_id=1,
        role=role,
        content=content,
        model=None,
        created_at="t",
    )


PROJECT = Project(
    id=1,
    name="AutoMarketPlus",
    description="Marketplace de vehículos",
    repo_path="C:/repos/auto",
    created_at="t",
)


def test_without_history_returns_system_and_question():
    messages = build_messages("SYS", "hola")

    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hola"},
    ]


def test_history_goes_between_system_and_question():
    history = [
        msg(1, "user", "p1"),
        msg(2, "assistant", "r1"),
    ]

    messages = build_messages("SYS", "p2", history=history)

    assert [m["content"] for m in messages] == [
        "SYS", "p1", "r1", "p2"
    ]


def test_ignores_tool_and_system_roles_in_history():
    history = [
        msg(1, "tool", "datos"),
        msg(2, "system", "otro"),
        msg(3, "user", "p1"),
    ]

    messages = build_messages("SYS", "p2", history=history)

    assert [m["content"] for m in messages] == [
        "SYS", "p1", "p2"
    ]


def test_drops_oldest_messages_when_over_budget():
    history = [
        msg(1, "user", "a" * 50),
        msg(2, "assistant", "b" * 50),
        msg(3, "user", "c" * 50),
    ]

    messages = build_messages(
        "SYS", "q", history=history, max_chars=110
    )

    contents = [m["content"] for m in messages]

    assert "a" * 50 not in contents
    assert "b" * 50 in contents
    assert "c" * 50 in contents
    assert contents[0] == "SYS"
    assert contents[-1] == "q"


def test_always_keeps_system_and_question_even_if_tiny_budget():
    history = [msg(1, "user", "x" * 100)]

    messages = build_messages(
        "SYS", "q", history=history, max_chars=1
    )

    assert [m["role"] for m in messages] == ["system", "user"]


def test_project_context_is_added_to_system_prompt():
    prompt = build_system_prompt("SYS", PROJECT)

    assert "AutoMarketPlus" in prompt
    assert "Marketplace de vehículos" in prompt
    assert "C:/repos/auto" in prompt
