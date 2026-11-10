"""Поиск по фрагментам: по векторам, по словам, гибрид. Фильтры и порог — на любом режиме."""
import argparse
import json
from typing import Literal

from app.core.config import settings
from app.core.db import close_pool, init_pool
from app.rag.embedder import embed_query
from app.repositories import chunks as chunks_repo

Mode = Literal["vector", "keyword", "hybrid"]


def search(
    query: str,
    *,
    mode: Mode = "hybrid",
    limit: int | None = None,
    min_score: float | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Возвращает фрагменты с идентификатором, текстом, оценкой и метаданными.

    Оценка зависит от режима: близость (vector), ранг текстового поиска (keyword),
    сумма вкладов RRF (hybrid) — сравнивать оценки разных режимов нельзя.
    """
    limit = limit or settings.search_limit
    min_score = settings.search_min_score if min_score is None else min_score

    if mode == "keyword":
        return chunks_repo.search_keyword(query, limit=limit, filters=filters)

    vector = embed_query(query)
    if mode == "vector":
        return chunks_repo.search_vector(vector, limit=limit, min_score=min_score, filters=filters)
    return chunks_repo.search_hybrid(
        vector,
        query,
        limit=limit,
        min_score=min_score,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
        filters=filters,
    )


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
            filters={"document": args.document, "article": args.article, "brand": args.brand},
        )
    finally:
        close_pool()
    for hit in hits:
        hit["text"] = hit["text"][:200]
    print(json.dumps(hits, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
