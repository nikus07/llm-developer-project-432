"""Раннер миграций: SQL-файлы из db/migrations применяются по порядку имён.

Запуск: python -m app.core.migrate
Применённые версии хранятся в таблице schema_migrations (её создаёт 0001_init.sql).
"""
import time
from pathlib import Path

import psycopg

from app.core.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def discover(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(directory.glob("*.sql"), key=lambda p: p.name)


def applied_versions(conn: psycopg.Connection) -> set[str]:
    # до первой миграции таблицы ещё нет — значит, применённых нет
    if conn.execute("SELECT to_regclass('schema_migrations')").fetchone()[0] is None:
        return set()
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def _connect(url: str, wait_s: float = 30.0) -> psycopg.Connection:
    """База в Compose поднимается не мгновенно: до wait_s секунд повторяем подключение."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return psycopg.connect(url)
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def migrate(database_url: str | None = None, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Применяет новые миграции, возвращает их версии. Каждая — в своей транзакции."""
    applied_now: list[str] = []
    with _connect(database_url or settings.database_url) as conn:
        done = applied_versions(conn)
        conn.commit()
        for path in discover(directory):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,)
                )
            applied_now.append(path.name)
    return applied_now


def main() -> None:
    applied = migrate()
    if not applied:
        print("Новых миграций нет")
        return
    for version in applied:
        print(f"Миграция применена: {version}")


if __name__ == "__main__":
    main()
