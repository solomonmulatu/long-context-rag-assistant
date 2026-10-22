import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    openai_api_key: str
    openai_chat_model: str="gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_directory: str = str(
        PROJECT_ROOT / "db_v1" / "chroma_db"
    )
    data_directory: str = "./data"
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()