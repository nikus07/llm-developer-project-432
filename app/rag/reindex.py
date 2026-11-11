"""Досчёт векторов: фрагментам, у которых их нет.

Запуск: python -m app.rag.reindex
Нужна после сбоя и после смены модели (смена модели — очистить колонку embedding).
Повторный запуск ничего не пересчитывает: берутся только фрагменты без вектора.
"""
from app.core.config import settings
from app.core.db import close_pool, init_pool
from app.core.logging import get_logger, setup_logging
from app.rag.embedder import embed_documents
from app.repositories import chunks as chunks_repo

log = get_logger(__name__)


def index_chunks(document_id: str | None = None, batch_size: int | None = None) -> int:
    """Считает и записывает векторы пачками. Возвращает число проиндексированных фрагментов."""
    batch_size = batch_size or settings.embedding_batch_size
    total = chunks_repo.count_without_embeddings()
    done = 0
    while True:
        batch = chunks_repo.list_without_embeddings(batch_size, document_id)
        if not batch:
            break
        vectors = embed_documents([row["text"] for row in batch])
        pairs = zip((row["chunk_id"] for row in batch), vectors, strict=True)
        chunks_repo.write_embeddings(list(pairs))
        done += len(batch)
        log.info("reindex_progress", done=done, total=total)
    if done == 0:
        log.info("reindex_nothing_to_do")
    return done


def main() -> None:
    setup_logging()
    init_pool()
    try:
        done = index_chunks()
    finally:
        close_pool()
    print(f"Проиндексировано фрагментов: {done}" if done else "Новых фрагментов нет")


if __name__ == "__main__":
    main()
