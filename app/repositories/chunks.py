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
