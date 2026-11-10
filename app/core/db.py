"""Пул соединений: ресурс уровня приложения.

Пул создаётся на старте процесса и закрывается при остановке. Операция берёт
соединение через connection() и возвращает его в пул — на одну операцию,
а не на весь шаг процесса.
"""
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import settings

_pool: ConnectionPool | None = None


def init_pool() -> ConnectionPool:
    """Создаёт пул. База может быть недоступна — соединения открываются лениво."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.db_pool_max_size,
            kwargs={"autocommit": True},
            open=False,
        )
        _pool.open(wait=False)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("пул соединений не создан: вызовите init_pool() на старте")
    return _pool


@contextmanager
def connection(timeout: float | None = None) -> Iterator[psycopg.Connection]:
    """Соединение из пула на одну операцию; возвращается при выходе из блока."""
    with pool().connection(timeout=timeout) as conn:
        yield conn
