
import json

import pytest

from app.tools.registry import (
    execute_tool,
    tool_definitions,
)

def test_tools_exposed():
    names = {
        tool["function"]["name"]
        for tool in tool_definitions()
    }

    assert names == {
        "lookup_ai_concept",
        "calculate_study_hours",
        "create_study_task",
        "estimate_learning_effort",
    }


def test_calculation():
    result = execute_tool(
        "calculate_study_hours",
        json.dumps({
            "hours_per_day": 1.5,
            "days": 4,
        }),
        False,
    )

    assert (
        result["data"]["total_hours"]
        == 6
    )


def test_invalid_args():
    invalid_range = execute_tool(
        "calculate_study_hours",
        json.dumps({
            "hours_per_day": 50,
            "days": 2,
        }),
        False,
    )

    assert (
        invalid_range["error"]
        == "invalid_arguments"
    )

    invalid_json = execute_tool(
        "calculate_study_hours",
        "not-json",
        False,
    )

    assert (
        invalid_json["error"]
        == "invalid_arguments"
    )


def test_unknown_tool():
    result = execute_tool(
        "run_shell",
        "{}",
        True,
    )

    assert (
        result["error"]
        == "unknown_tool"
    )

def run_effort(args: dict) -> dict:
    return execute_tool(
        "estimate_learning_effort",
        json.dumps(args),
        False,
    )


# ---------- casos válidos ----------

def test_effort_example_from_teacher():
    result = run_effort({
        "difficulty": "intermedio",
        "topics": 5,
        "weekly_hours": 5,
    })

    assert result["ok"] is True
    assert result["data"]["total_hours"] == 20
    assert result["data"]["weeks_needed"] == 4


def test_effort_rounds_weeks_up():
    result = run_effort({
        "difficulty": "basico",
        "topics": 5,
        "weekly_hours": 4,
    })

    assert result["data"]["total_hours"] == 10
    assert result["data"]["weeks_needed"] == 3


# ---------- valores límite válidos ----------

def test_effort_minimum_values():
    result = run_effort({
        "difficulty": "basico",
        "topics": 1,
        "weekly_hours": 0.01,
    })

    assert result["ok"] is True


def test_effort_maximum_values():
    result = run_effort({
        "difficulty": "avanzado",
        "topics": 50,
        "weekly_hours": 60,
    })

    assert result["ok"] is True
    assert result["data"]["total_hours"] == 300
    assert result["data"]["weeks_needed"] == 5


# ---------- valores límite inválidos ----------

@pytest.mark.parametrize(
    "args",
    [
        {"difficulty": "basico", "topics": 0, "weekly_hours": 5},
        {"difficulty": "basico", "topics": 51, "weekly_hours": 5},
        {"difficulty": "basico", "topics": 5, "weekly_hours": 0},
        {"difficulty": "basico", "topics": 5, "weekly_hours": 60.01},
    ],
)
def test_effort_boundary_invalid(args):
    assert run_effort(args)["error"] == "invalid_arguments"


# ---------- argumentos inválidos ----------

@pytest.mark.parametrize(
    "args",
    [
        {"difficulty": "experto", "topics": 5, "weekly_hours": 5},
        {"topics": 5, "weekly_hours": 5},
        {"difficulty": "basico", "topics": "cinco", "weekly_hours": 5},
        {
            "difficulty": "basico",
            "topics": 5,
            "weekly_hours": 5,
            "hours_per_topic": 4,
        },
    ],
)
def test_effort_invalid_arguments(args):
    assert run_effort(args)["error"] == "invalid_arguments"


def test_effort_invalid_json():
    result = execute_tool("estimate_learning_effort", "not-json", False)

    assert result["error"] == "invalid_arguments"


# ---------- consistencia contrato y tabla ----------

def test_every_difficulty_has_hours():
    for level in ("basico", "intermedio", "avanzado"):
        result = run_effort({
            "difficulty": level,
            "topics": 1,
            "weekly_hours": 5,
        })

        assert result["ok"] is True
