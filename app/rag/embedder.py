"""Локальные эмбеддинги: модель грузится один раз, векторы считаются пачкой.

Модели семейства e5 обучены с задачными префиксами: запрос и документ кодируются
по-разному («query: » / «passage: »). Префиксы — часть контракта модели, они лежат
в настройках и берутся из её карточки.
"""
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _model():
    # импорт внутри: тесты и процессы, которым модель не нужна, не тянут torch
    from sentence_transformers import SentenceTransformer

    log.info("embedding_model_loading", model=settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    dim = model.get_embedding_dimension()
    if dim != settings.embed_dim:
        raise RuntimeError(
            f"размерность модели {dim} не совпадает с embed_dim={settings.embed_dim}"
            " (и с колонкой vector в миграции)"
        )
    return model


def _encode(texts: list[str], prefix: str) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(
        [prefix + text for text in texts],
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,  # косинус = скалярное произведение
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Фрагменты документов, пачкой."""
    return _encode(texts, settings.embedding_document_prefix)


def embed_query(text: str) -> list[float]:
    """Поисковый запрос: префикс другой, чем у документа."""
    return _encode([text], settings.embedding_query_prefix)[0]
