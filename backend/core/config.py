"""
config.py — Centralised application settings using Pydantic BaseSettings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Absolute path to .env — works regardless of where uvicorn is launched from
ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):

    # OpenAI
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_whisper_model: str = "whisper-1"

    # Pinecone
    pinecone_api_key: str
    pinecone_index: str = "kust-docs"

    # Auth
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Application
    app_name: str = "KUST AI Assistant"
    debug: bool = False
    pdf_dir: str = "./pdfs"

    # CORS
    allowed_origins: str = "http://localhost:3000"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = str(ENV_FILE)
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()