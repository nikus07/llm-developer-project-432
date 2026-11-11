import uuid
from decimal import Decimal

import psycopg
import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import db
from app.core.config import settings
from app.core.migrate import migrate
from app.core.tracing import RequestContextMiddleware, bind_job, current_request_id
from app.repositories import llm_calls as llm_calls_repo
from app.temporal.tracing import ids_from_workflow_id


def _app() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping():
        bind_job("job-42")  # как после создания задачи в обработчике
        return {
            "request_id": current_request_id(),
            "context": structlog.contextvars.get_contextvars(),
        }

    return TestClient(app)


def test_request_id_is_born_at_the_entrance_and_returned():
    response = _app().get("/ping")
    request_id = response.headers["x-request-id"]
    assert request_id and response.json()["request_id"] == request_id


def test_request_id_from_header_is_kept():
    response = _app().get("/ping", headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
    assert response.json()["request_id"] == "abc123"


def test_each_request_gets_its_own_id_and_job_does_not_leak():
    client = _app()
    first = client.get("/ping").json()
    second = client.get("/ping").json()
    assert first["request_id"] != second["request_id"]
    assert second["context"] == {"request_id": second["request_id"], "job_id": "job-42"}


def test_job_id_is_taken_from_workflow_id():
    assert ids_from_workflow_id("card-abc123") == {"job_id": "abc123"}
    assert ids_from_workflow_id("doc-77") == {"document_id": "77"}
    assert ids_from_workflow_id("test-xyz") == {}
    assert ids_from_workflow_id("nodash") == {}


# --- таблица вызовов на живой базе ---


@pytest.fixture(scope="module")
def live_db():
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            pass
    except psycopg.OperationalError:
        pytest.skip("база недоступна: поднимите make infra")
    migrate()
    db.init_pool()
    yield
    with db.connection() as conn:  # тестовые вызовы не должны попасть в общий отчёт
        conn.execute("DELETE FROM llm_calls WHERE job_id LIKE 'test-%'")
    db.close_pool()


def _call(job_id, model, tier, tokens_in, tokens_out, cost, status="ok"):
    llm_calls_repo.insert(
        job_id=job_id, request_id="req", agent="Test", model=model, tier=tier,
        input_tokens=tokens_in, output_tokens=tokens_out, cost=Decimal(cost),
        duration_ms=100, status=status,
    )


def test_cost_of_one_job_is_one_query_and_exact(live_db):
    job_id = f"test-{uuid.uuid4().hex}"
    _call(job_id, "m-main", "main", 1000, 500, "0.045000")
    _call(job_id, "m-light", "light", 2000, 100, "0.000007")
    _call(job_id, "m-main", "main", 5, 5, "0.000001")
    _call(job_id, "m-main", "main", 999, 999, "9.999999", status="error")  # ошибка не в счёт

    total = llm_calls_repo.cost_for_job(job_id)

    assert total["calls"] == 3
    assert total["input_tokens"] == 3005
    assert total["cost"] == Decimal("0.045008")  # без накопленной ошибки float
    assert isinstance(total["cost"], Decimal)
    assert llm_calls_repo.cost_for_job("нет-такой")["calls"] == 0


def test_breakdown_by_model_separates_light_and_main(live_db):
    job_id = f"test-{uuid.uuid4().hex}"
    model_main, model_light = f"main-{job_id[-6:]}", f"light-{job_id[-6:]}"
    for _ in range(3):
        _call(job_id, model_main, "main", 1000, 500, "0.045000")
    for _ in range(2):
        _call(job_id, model_light, "light", 1000, 500, "0.009000")

    rows = {r["model"]: r for r in llm_calls_repo.breakdown_by_model()}

    assert rows[model_main]["calls"] == 3 and rows[model_main]["cost"] == Decimal("0.135000")
    assert rows[model_light]["calls"] == 2 and rows[model_light]["cost"] == Decimal("0.018000")
    assert rows[model_main]["tier"] == "main" and rows[model_light]["tier"] == "light"
