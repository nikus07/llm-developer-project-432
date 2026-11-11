"""Таблица chunks: фрагменты документов, векторы и поиск.

Фрагмент — словарь {"chunk_id", "doc_id", "text", "metadata", "filename", "score"}.
Функции поиска и чтения по идентификаторам получают соединение первым аргументом.
"""
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.db import connection
from app.rag.embedder import embed_query

# --- запись и чтение ----------------------------------------------------------------


def replace_for_document(doc_id: str, chunks: list[dict]) -> None:
    """Заменяет фрагменты документа целиком: повторный разбор не плодит дублей."""
    with connection() as conn, conn.transaction():
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, doc_id, ordinal, text, metadata) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (c["chunk_id"], doc_id, c["ordinal"], c["text"], Jsonb(c["metadata"]))
                    for c in chunks
                ],
            )


def list_for_document(doc_id: str) -> list[dict]:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.text, c.metadata, d.filename
              FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.doc_id = %s ORDER BY c.ordinal
            """,
            (doc_id,),
        )
        return cur.fetchall()


def get_chunks_by_ids(conn, chunk_ids: list[str]) -> dict[str, dict]:
    """Фрагменты по идентификаторам: проверка, что ссылка указывает на существующий."""
    if not chunk_ids:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id AS chunk_id, c.doc_id, c.text, c.metadata, d.filename
              FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.id = ANY(%s)
            """,
            (list(chunk_ids),),
        )
        return {row["chunk_id"]: row for row in cur.fetchall()}


# --- векторы ------------------------------------------------------------------------


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def count_without_embeddings() -> int:
    with connection() as conn:
        return conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]


def list_without_embeddings(limit: int, doc_id: str | None = None) -> list[dict]:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id AS chunk_id, text FROM chunks
             WHERE embedding IS NULL AND (%s::text IS NULL OR doc_id = %s)
             ORDER BY doc_id, ordinal LIMIT %s
            """,
            (doc_id, doc_id, limit),
        )
        return cur.fetchall()


def write_embeddings(rows: list[tuple[str, list[float]]]) -> None:
    """Векторы пачкой, одной транзакцией."""
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(vector_literal(vector), chunk_id) for chunk_id, vector in rows],
        )


# --- поиск --------------------------------------------------------------------------

_FILTER_SQL = {
    "document": "d.filename = %({name})s",
    "section": "c.metadata ->> 'section' = %({name})s",
    "page": "c.metadata ->> 'page' = %({name})s::text",
    "articul": "c.metadata ->> 'articul' ILIKE '%%' || %({name})s || '%%'",
    "brand": "c.metadata ->> 'brand' ILIKE '%%' || %({name})s || '%%'",
}

_HIT_COLUMNS = "c.id AS chunk_id, c.doc_id, c.text, c.metadata, d.filename"


def _where(doc_ids: list[str] | None, filters: dict | None) -> tuple[str, dict]:
    """Условия по документам и метаданным: только известные ключи, значения — параметрами."""
    clauses: list[str] = []
    params: dict = {}
    if doc_ids is not None:
        clauses.append("c.doc_id = ANY(%(f_doc_ids)s)")
        params["f_doc_ids"] = list(doc_ids)
    for name, value in (filters or {}).items():
        if value is None or name not in _FILTER_SQL:
            continue
        key = f"f_{name}"
        clauses.append(_FILTER_SQL[name].format(name=key))
        params[key] = value
    return "".join(f" AND {clause}" for clause in clauses), params


def _threshold(value: float | None) -> float:
    return settings.search_min_score if value is None else value


def search_vector(
    conn,
    query: str,
    top_k: int = 5,
    threshold: float | None = None,
    doc_ids: list[str] | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Ближайшие по косинусу; оценка — близость (1 − расстояние), ниже порога не отдаём."""
    where, params = _where(doc_ids, filters)
    params |= {
        "q": vector_literal(embed_query(query)),
        "limit": top_k,
        "min_score": _threshold(threshold),
    }
    sql = f"""
        SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %(q)s::vector) AS score
          FROM chunks c JOIN documents d ON d.id = c.doc_id
         WHERE c.embedding IS NOT NULL
           AND 1 - (c.embedding <=> %(q)s::vector) >= %(min_score)s {where}
         ORDER BY c.embedding <=> %(q)s::vector
         LIMIT %(limit)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_fts(
    conn,
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Обычное ранжирование полнотекстового поиска: слова запроса должны быть в тексте."""
    where, params = _where(doc_ids, filters)
    params |= {"text": query, "limit": top_k}
    sql = f"""
        SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
          FROM chunks c JOIN documents d ON d.id = c.doc_id,
               plainto_tsquery('russian', %(text)s) AS q(query)
         WHERE c.tsv @@ q.query {where}
         ORDER BY score DESC, c.id
         LIMIT %(limit)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_hybrid(
    conn,
    query: str,
    top_k: int = 5,
    threshold: float | None = None,
    doc_ids: list[str] | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Слияние по обратным позициям (RRF): вклад каждого списка — 1 / (k + позиция).

    Фрагмент, найденный обоими списками, получает два вклада и оказывается выше.
    Порог действует на векторный список: мусор до слияния не доходит.
    """
    where, params = _where(doc_ids, filters)
    params |= {
        "q": vector_literal(embed_query(query)),
        "text": query,
        "limit": top_k,
        "min_score": _threshold(threshold),
        "candidates": settings.search_candidates,
        "k": settings.rrf_k,
    }
    sql = f"""
        WITH vec AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rnk
              FROM (
                  SELECT c.id, c.embedding <=> %(q)s::vector AS dist
                    FROM chunks c JOIN documents d ON d.id = c.doc_id
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
                    FROM chunks c JOIN documents d ON d.id = c.doc_id,
                         plainto_tsquery('russian', %(text)s) AS tq(query)
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
          FROM fused f JOIN chunks c ON c.id = f.id JOIN documents d ON d.id = c.doc_id
         ORDER BY f.score DESC, c.id
         LIMIT %(limit)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
