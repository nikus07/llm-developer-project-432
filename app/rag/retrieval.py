"""Поиск по фрагментам: по векторам, по словам, гибрид. Фильтры и порог — на любом режиме."""
import argparse
import json
from typing import Literal

from app.core.config import settings
from app.core.db import close_pool, connection, init_pool
from app.repositories import chunks as chunks_repo

Mode = Literal["vector", "keyword", "hybrid"]


def search(
    query: str,
    *,
    mode: Mode = "hybrid",
    limit: int | None = None,
    min_score: float | None = None,
    filters: dict | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Возвращает фрагменты с идентификатором, текстом, оценкой и метаданными.

    Оценка зависит от режима: близость (vector), ранг текстового поиска (keyword),
    сумма вкладов RRF (hybrid) — сравнивать оценки разных режимов нельзя.
    """
    limit = limit or settings.search_limit
    filters = dict(filters or {})
    document_id = filters.pop("document_id", None)
    if document_id:
        doc_ids = [*(doc_ids or []), document_id]
    with connection() as conn:
        if mode == "keyword":
            return chunks_repo.search_fts(
                conn, query, top_k=limit, doc_ids=doc_ids, filters=filters
            )
        if mode == "vector":
            return chunks_repo.search_vector(
                conn, query, top_k=limit, threshold=min_score, doc_ids=doc_ids, filters=filters
            )
        return chunks_repo.search_hybrid(
            conn, query, top_k=limit, threshold=min_score, doc_ids=doc_ids, filters=filters
        )


def retrieve_chunks(query: str, doc_ids: list[str] | None = None) -> list[dict]:
    """Фрагменты для генерации: основной запрос и по запросу на каждый аспект карточки."""
    queries = [query, *(f"{query} {aspect}" for aspect in settings.generate_aspects)]
    hits: list[dict] = []
    seen: set[str] = set()
    with connection() as conn:
        for text in queries:
            found = chunks_repo.search_hybrid(
                conn,
                text,
                top_k=settings.generate_top_k,
                threshold=settings.generate_min_score,
                doc_ids=doc_ids,
            )
            for hit in found:
                if hit["chunk_id"] not in seen:
                    seen.add(hit["chunk_id"])
                    hits.append(hit)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск по фрагментам документов")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--document")
    parser.add_argument("--article")
    parser.add_argument("--brand")
    args = parser.parse_args()

    init_pool()
    try:
        hits = search(
            args.query,
            mode=args.mode,
            limit=args.limit,
            min_score=args.min_score,
            filters={"document": args.document, "articul": args.article, "brand": args.brand},
        )
    finally:
        close_pool()
    for hit in hits:
        hit["text"] = hit["text"][:200]
    print(json.dumps(hits, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
