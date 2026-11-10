"""Таблица jobs: весь SQL по задачам живёт здесь."""
import uuid

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.db import connection

_COLUMNS = "id, status, attempts, result, error"


def create(payload: dict, idempotency_key: str | None = None) -> tuple[dict, bool]:
    """Создаёт задачу. Тот же ключ идемпотентности возвращает первую задачу.

    Возвращает (задача, создана_ли_сейчас).
    """
    job_id = uuid.uuid4().hex
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            INSERT INTO jobs (id, idempotency_key, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING {_COLUMNS}
            """,
            (job_id, idempotency_key, Jsonb(payload)),
        )
        row = cur.fetchone()
        if row is not None:
            return row, True
        cur.execute(
            f"SELECT {_COLUMNS} FROM jobs WHERE idempotency_key = %s", (idempotency_key,)
        )
        return cur.fetchone(), False


def get(job_id: str) -> dict | None:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def update_status(
    job_id: str,
    status: str,
    *,
    attempts: int | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Короткое обращение к базе: соединение берётся на одну операцию."""
    with connection() as conn:
        conn.execute(
            """
            UPDATE jobs
               SET status     = %s,
                   attempts   = COALESCE(%s, attempts),
                   result     = COALESCE(%s, result),
                   error      = COALESCE(%s, error),
                   updated_at = now()
             WHERE id = %s
            """,
            (
                status,
                attempts,
                Jsonb(result) if result is not None else None,
                error,
                job_id,
            ),
        )


def get_payload(job_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute("SELECT payload FROM jobs WHERE id = %s", (job_id,)).fetchone()
    return row[0] if row else None
