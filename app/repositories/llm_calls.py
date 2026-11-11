"""Таблица llm_calls: учёт вызовов модели и стоимости."""
from decimal import Decimal

from psycopg.rows import dict_row

from app.core.db import connection


def insert(
    *,
    job_id: str | None,
    request_id: str | None,
    agent: str,
    model: str,
    tier: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    duration_ms: int,
    status: str = "ok",
    error: str | None = None,
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO llm_calls
                (job_id, request_id, agent, model, tier, input_tokens, output_tokens,
                 cost, duration_ms, status, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (job_id, request_id, agent, model, tier, input_tokens, output_tokens, cost,
             duration_ms, status, error),
        )


def breakdown_by_model() -> list[dict]:
    """Разбивка по моделям: сколько вызовов и на какую сумму ушло на каждую."""
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT model, tier,
                   count(*)                       AS calls,
                   sum(input_tokens)::bigint      AS input_tokens,
                   sum(output_tokens)::bigint     AS output_tokens,
                   sum(cost)                      AS cost,
                   avg(duration_ms)::int          AS avg_ms
              FROM llm_calls
             WHERE status = 'ok'
             GROUP BY model, tier
             ORDER BY tier, model
            """
        )
        return cur.fetchall()


def cost_for_job(job_id: str) -> dict:
    """Стоимость одной задачи — одним запросом по её идентификатору."""
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT count(*)                                  AS calls,
                   coalesce(sum(input_tokens), 0)::bigint    AS input_tokens,
                   coalesce(sum(output_tokens), 0)::bigint   AS output_tokens,
                   coalesce(sum(cost), 0)                    AS cost,
                   coalesce(sum(duration_ms), 0)::bigint     AS duration_ms
              FROM llm_calls
             WHERE job_id = %s AND status = 'ok'
            """,
            (job_id,),
        )
        return cur.fetchone()


def calls_for_job(job_id: str) -> list[dict]:
    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT created_at, agent, model, tier, input_tokens, output_tokens, cost,
                   duration_ms, status
              FROM llm_calls WHERE job_id = %s ORDER BY id
            """,
            (job_id,),
        )
        return cur.fetchall()
