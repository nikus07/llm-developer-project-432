from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "gemma-4-e4b-it"

    database_url: str = "postgresql://card:card@localhost:5432/card"
    rabbitmq_url: str = "amqp://card:card@localhost:5672//"

    temporal_host: str = "localhost:7233"
    temporal_task_queue: str = "card-queue"

    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3

    db_pool_max_size: int = 10

    card_title_max_len: int = 60
    # ниже порога карточка уходит человеку на проверку, а не в «готово»
    confidence_threshold: float = 0.6
    # сколько раз можно вернуть модели невалидный ответ на лечение
    structured_max_repairs: int = 2

    chunk_size: int = 800
    chunk_overlap: int = 150
    max_upload_mb: int = 10
    upload_dir: Path = Path(__file__).resolve().parents[2] / "uploads"


settings = Settings()
