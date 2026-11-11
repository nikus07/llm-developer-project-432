"""Пул соединений: ресурс уровня приложения.

Пул создаётся при первом обращении (или явно через init_pool() на старте процесса) и
закрывается при остановке. Операция берёт соединение через connection() и возвращает
его в пул — на одну операцию, а не на весь шаг процесса.
"""
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import settings

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def init_pool() -> ConnectionPool:
    """Создаёт пул. База может быть недоступна: соединения открываются лениво."""
    global _pool
    with _lock:
        if _pool is None:
            pool_ = ConnectionPool(
                settings.database_url,
                min_size=1,
                max_size=settings.db_pool_max_size,
                kwargs={"autocommit": True},
                open=False,
            )
            pool_.open(wait=False)
            _pool = pool_
    return _pool


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def pool() -> ConnectionPool:
    """Пул приложения; создаётся при первом обращении, если его ещё нет."""
    return _pool if _pool is not None else init_pool()


@contextmanager
def connection(timeout: float | None = None) -> Iterator[psycopg.Connection]:
    """Соединение из пула на одну операцию; возвращается при выходе из блока."""
    with pool().connection(timeout=timeout) as conn:
        yield conn
