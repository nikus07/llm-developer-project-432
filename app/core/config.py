from decimal import Decimal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "gemma-4-e4b-it"
    # служебная (дешёвая) модель: проверка черновика, правка поля, судья; пусто — как основная
    llm_model_light: str | None = None
    # цены, ₽ за миллион токенов. Локальная модель бесплатна, поэтому цены условные:
    # они нужны, чтобы таблица вызовов и разбивка по моделям показывали разницу тарифов
    llm_price_main_input: Decimal = Decimal("15")
    llm_price_main_output: Decimal = Decimal("60")
    llm_price_light_input: Decimal = Decimal("3")
    llm_price_light_output: Decimal = Decimal("12")

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
    # локальные эмбеддинги: модель, размерность (должна совпадать с миграцией 0004) и
    # префиксы из карточки модели — без них близость релевантных пар проседает
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    embedding_query_prefix: str = "query: "
    embedding_document_prefix: str = "passage: "
    embedding_batch_size: int = 32

    # поиск: порог косинусной близости, размер выдачи, число кандидатов из каждого списка
    # и константа слияния по обратным позициям (RRF)
    search_min_score: float = 0.805
    search_limit: int = 5
    search_candidates: int = 20
    rrf_k: int = 60

    # генерация по документам: предел размера контекста, число фрагментов на запрос
    # и аспекты, по которым ищем сведения о товаре. Поиск идёт внутри выбранных документов,
    # поэтому порог близости здесь 0: всё найденное в них — не мусор, а материал.
    context_max_chars: int = 6000
    generate_top_k: int = 8
    generate_min_score: float = 0.0
    generate_aspects: list[str] = [
        "технические характеристики",
        "артикул модель",
        "гарантия комплект поставки",
        "цена условия поставки",
    ]

    # защита от инъекций: столько подозрительных фрагментов — документ целиком к человеку
    injection_max_suspicious: int = 3

    upload_dir: Path = Path(__file__).resolve().parents[2] / "uploads"


settings = Settings()
