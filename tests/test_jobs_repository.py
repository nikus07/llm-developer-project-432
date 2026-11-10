"""Репозиторий задач на живой базе (Compose). Без базы тесты пропускаются."""
import psycopg
import pytest

from app.core import db
from app.core.config import settings
from app.core.migrate import migrate
from app.repositories import jobs as jobs_repo


@pytest.fixture(scope="module", autouse=True)
def live_db():
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            pass
    except psycopg.OperationalError:
        pytest.skip("база недоступна: поднимите make infra")
    migrate()
    db.init_pool()
    yield
    db.close_pool()


def test_create_and_get():
    job, created = jobs_repo.create({"supplier_text": "текст"}, idempotency_key=None)
    assert created is True
    assert job["status"] == "pending"
    assert job["attempts"] == 0
    assert jobs_repo.get(job["id"])["id"] == job["id"]
    assert jobs_repo.get_payload(job["id"]) == {"supplier_text": "текст"}


def test_same_idempotency_key_returns_first_job():
    key = "test-key-same"
    with db.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE idempotency_key = %s", (key,))
    first, created_first = jobs_repo.create({"supplier_text": "a"}, key)
    second, created_second = jobs_repo.create({"supplier_text": "b"}, key)
    assert created_first is True
    assert created_second is False
    assert second["id"] == first["id"]


def test_update_status_keeps_unset_fields():
    job, _ = jobs_repo.create({"supplier_text": "x"})
    draft = {
        "title": "t",
        "description": "d",
        "characteristics": {},
        "benefits": [],
    }
    jobs_repo.update_status(job["id"], "awaiting_confirmation", attempts=2, result=draft)
    jobs_repo.update_status(job["id"], "approved")
    row = jobs_repo.get(job["id"])
    assert row["status"] == "approved"
    assert row["attempts"] == 2
    assert row["result"] == draft


def test_get_unknown_returns_none():
    assert jobs_repo.get("нет-такой") is None
