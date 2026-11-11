"""Таблица documents: весь SQL по документам живёт здесь."""
import uuid

from psycopg.rows import dict_row

from app.core.db import connection

_SELECT = """
    SELECT d.id, d.filename, d.kind, d.status, d.reason, d.pages,
           (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks
      FROM documents d
"""


def create(filename: str, content_hash: str, size_bytes: int, kind: str) -> tuple[dict, bool]:
    """Создаёт документ. Тот же файл (по хэшу) возвращает первую запись.

    Возвращает (документ, создан_ли_сейчас).
    """
    doc_id = uuid.uuid4().hex
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO documents (id, filename, content_hash, size_bytes, kind)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (doc_id, filename, content_hash, size_bytes, kind),
        )
        created = cur.fetchone() is not None
        cur.execute(_SELECT + " WHERE d.content_hash = %s", (content_hash,))
        return cur.fetchone(), created


def get(document_id: str) -> dict | None:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SELECT + " WHERE d.id = %s", (document_id,))
        return cur.fetchone()


def set_status(
    document_id: str, status: str, *, reason: str | None = None, pages: int | None = None
) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE documents
               SET status     = %s,
                   reason     = %s,
                   pages      = COALESCE(%s, pages),
                   updated_at = now()
             WHERE id = %s
            """,
            (status, reason, pages, document_id),
        )
