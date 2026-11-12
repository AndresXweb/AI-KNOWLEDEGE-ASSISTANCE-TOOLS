from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ConceptArgs(StrictArgs):
    concept: str = Field(
        min_length=2,
        max_length=80,
    )

class EffortArgs(StrictArgs):
    difficulty: Literal["basico", "intermedio", "avanzado"] = Field(
        description=(
            "Nivel de dificultad de la actividad formativa: "
            "basico, intermedio o avanzado."
        ),
    )

    topics: int = Field(
        ge=1,
        le=50,
        description="Cantidad de temas que hay que estudiar.",
    )

    weekly_hours: float = Field(
        gt=0,
        le=60,
        description="Horas que el aprendiz puede estudiar por semana.",
    )

class HoursArgs(StrictArgs):
    hours_per_day: float = Field(
        gt=0,
        le=12,
    )

    days: int = Field(
        ge=1,
        le=31,
    )


class TaskArgs(StrictArgs):
    title: str = Field(
        min_length=3,
        max_length=120,
    )

    estimated_hours: float = Field(
        gt=0,
        le=40,
    )
