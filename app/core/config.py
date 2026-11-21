import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    llm_provider: str = "openrouter"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434/v1"
    memory_db_path: str = "data/memory.db"

    @property
    def active_model(self) -> str:
        if self.llm_provider.strip().lower() == "ollama":
            return self.ollama_model

        return self.openrouter_model


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.getenv(
            "OPENROUTER_API_KEY", ""
        ),
        openrouter_model=os.getenv(
            "OPENROUTER_MODEL", "openrouter/free"
        ),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        llm_provider=os.getenv(
            "LLM_PROVIDER", "openrouter"
        ),
        ollama_model=os.getenv(
            "OLLAMA_MODEL", "llama3.1:8b"
        ),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434/v1",
        ),
        memory_db_path=os.getenv(
            "MEMORY_DB_PATH", "data/memory.db"
        ),
    )
