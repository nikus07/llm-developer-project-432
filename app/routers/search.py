from typing import Annotated

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from app.rag.retrieval import Mode, search
from app.schemas.search import SearchHit

router = APIRouter(tags=["search"])


def _hit(row: dict) -> dict:
    meta = row.get("metadata") or {}
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "document": row["filename"],
        "page": meta.get("page"),
        "section": meta.get("section"),
        "kind": meta.get("kind"),
        "articul": meta.get("articul"),
        "brand": meta.get("brand"),
        "text": row["text"],
        "score": float(row["score"]),
    }


@router.get("/search", response_model=list[SearchHit])
async def search_chunks(
    q: Annotated[str, Query(min_length=1, description="Поисковый запрос")],
    mode: Mode = "hybrid",
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
    min_score: Annotated[float | None, Query(ge=0, le=1)] = None,
    document: str | None = None,
    document_id: str | None = None,
    article: str | None = None,
    brand: str | None = None,
    page: int | None = None,
) -> list[dict]:
    """Поиск по фрагментам: по смыслу (vector), по словам (keyword) или гибрид."""
    filters = {
        "document": document,
        "document_id": document_id,
        "articul": article,
        "brand": brand,
        "page": page,
    }
    # эмбеддинг запроса и база блокируют — выносим из цикла событий
    rows = await run_in_threadpool(
        search, q, mode=mode, limit=limit, min_score=min_score, filters=filters
    )
    return [_hit(row) for row in rows]
