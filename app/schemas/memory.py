from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    repo_path: str | None = Field(default=None, max_length=500)


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str
    repo_path: str | None
    created_at: str


class ConversationOut(BaseModel):
    id: int
    project_id: int | None
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: str | None
    created_at: str
