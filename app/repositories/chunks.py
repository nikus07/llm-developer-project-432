"""Таблица chunks: фрагменты документов."""
from psycopg.rows import dict_row

from app.core.db import connection
from app.parsers.models import Chunk


def replace_for_document(document_id: str, document: str, chunks: list[Chunk]) -> None:
    """Заменяет фрагменты документа целиком: повторный разбор не плодит дублей."""
    with connection() as conn, conn.transaction():
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (document_id, document, idx, kind, text, page, section, article, brand)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        document_id,
                        document,
                        c.idx,
                        c.kind,
                        c.text,
                        c.page,
                        c.section,
                        c.article,
                        c.brand,
                    )
                    for c in chunks
                ],
            )


def list_for_document(document_id: str) -> list[dict]:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT idx, kind, text, page, section, article, brand, document
              FROM chunks WHERE document_id = %s ORDER BY idx
            """,
            (document_id,),
        )
        return cur.fetchall()


# --- векторы и поиск ---------------------------------------------------------------

_FILTER_COLUMNS = {
    "document": "c.document = %({name})s",
    "document_id": "c.document_id = %({name})s",
    "section": "c.section = %({name})s",
    "page": "c.page = %({name})s",
    "article": "c.article ILIKE '%%' || %({name})s || '%%'",
    "brand": "c.brand ILIKE '%%' || %({name})s || '%%'",
}

_HIT_COLUMNS = (
    "c.id, c.document_id, c.document, c.idx, c.kind, c.page, c.section, "
    "c.article, c.brand, c.text"
)


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def count_without_embeddings() -> int:
    with connection() as conn:
        return conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]


def list_without_embeddings(limit: int, document_id: str | None = None) -> list[dict]:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, text FROM chunks
             WHERE embedding IS NULL AND (%s::text IS NULL OR document_id = %s)
             ORDER BY id LIMIT %s
            """,
            (document_id, document_id, limit),
        )
        return cur.fetchall()


def write_embeddings(rows: list[tuple[int, list[float]]]) -> None:
    """Векторы пачкой, одной транзакцией."""
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(vector_literal(vector), chunk_id) for chunk_id, vector in rows],
        )


def _filters(filters: dict | None) -> tuple[str, dict]:
    """Условия по метаданным: только известные ключи, значения — параметрами."""
    clauses: list[str] = []
    params: dict = {}
    for name, value in (filters or {}).items():
        if value is None or name not in _FILTER_COLUMNS:
            continue
        key = f"f_{name}"
        clauses.append(_FILTER_COLUMNS[name].format(name=key))
        params[key] = value
    return "".join(f" AND {clause}" for clause in clauses), params


def search_vector(
    query_vector: list[float], *, limit: int, min_score: float, filters: dict | None = None
) -> list[dict]:
    """Ближайшие по косинусу; оценка — близость (1 − расстояние), ниже порога не отдаём."""
    where, params = _filters(filters)
    params |= {"q": vector_literal(query_vector), "limit": limit, "min_score": min_score}
    sql = f"""
        SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %(q)s::vector) AS score
          FROM chunks c
         WHERE c.embedding IS NOT NULL
           AND 1 - (c.embedding <=> %(q)s::vector) >= %(min_score)s {where}
         ORDER BY c.embedding <=> %(q)s::vector
         LIMIT %(limit)s
    """
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_keyword(query_text: str, *, limit: int, filters: dict | None = None) -> list[dict]:
    """Обычное ранжирование полнотекстового поиска: слова запроса должны быть в тексте."""
    where, params = _filters(filters)
    params |= {"text": query_text, "limit": limit}
    sql = f"""
        SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
          FROM chunks c, plainto_tsquery('russian', %(text)s) AS q(query)
         WHERE c.tsv @@ q.query {where}
         ORDER BY score DESC, c.id
         LIMIT %(limit)s
    """
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_hybrid(
    query_vector: list[float],
    query_text: str,
    *,
    limit: int,
    min_score: float,
    candidates: int,
    rrf_k: int,
    filters: dict | None = None,
) -> list[dict]:
    """Слияние по обратным позициям (RRF): вклад каждого списка — 1 / (k + позиция).

    Фрагмент, найденный обоими списками, получает два вклада и оказывается выше.
    Порог действует на векторный список: мусор до слияния не доходит.
    """
    where, params = _filters(filters)
    params |= {
        "q": vector_literal(query_vector),
        "text": query_text,
        "limit": limit,
        "min_score": min_score,
        "candidates": candidates,
        "k": rrf_k,
    }
    sql = f"""
        WITH vec AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rnk
              FROM (
                  SELECT c.id, c.embedding <=> %(q)s::vector AS dist
                    FROM chunks c
                   WHERE c.embedding IS NOT NULL
                     AND 1 - (c.embedding <=> %(q)s::vector) >= %(min_score)s {where}
                   ORDER BY dist
                   LIMIT %(candidates)s
              ) v
        ),
        kw AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY rank DESC, id) AS rnk
              FROM (
                  SELECT c.id, ts_rank_cd(c.tsv, tq.query) AS rank
                    FROM chunks c, plainto_tsquery('russian', %(text)s) AS tq(query)
                   WHERE c.tsv @@ tq.query {where}
                   ORDER BY rank DESC, c.id
                   LIMIT %(candidates)s
              ) k
        ),
        fused AS (
            SELECT id, SUM(1.0 / (%(k)s + rnk)) AS score
              FROM (SELECT id, rnk FROM vec UNION ALL SELECT id, rnk FROM kw) u
             GROUP BY id
        )
        SELECT {_HIT_COLUMNS}, f.score
          FROM fused f JOIN chunks c ON c.id = f.id
         ORDER BY f.score DESC, c.id
         LIMIT %(limit)s
    """
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
