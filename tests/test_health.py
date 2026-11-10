"""Живость и готовность: без базы и без сети."""
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.main import app
from app.routers import health

client = TestClient(app)  # без with: lifespan (и пул) не запускаются


def test_live_does_not_touch_database(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("живость не должна ходить в базу")

    monkeypatch.setattr(health, "connection", boom)
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_fails_when_database_is_down(monkeypatch):
    @contextmanager
    def down(timeout=None):
        raise ConnectionError("база остановлена")
        yield

    monkeypatch.setattr(health, "connection", down)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["detail"]["database"] == "down"


def test_ready_reports_vector_extension(monkeypatch):
    class FakeConn:
        def execute(self, sql):
            class Cur:
                @staticmethod
                def fetchone():
                    return ("0.8.0",)

            return Cur()

    @contextmanager
    def ok(timeout=None):
        yield FakeConn()

    monkeypatch.setattr(health, "connection", ok)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "up", "vector": "0.8.0"}


def test_ready_fails_without_vector_extension(monkeypatch):
    class FakeConn:
        def execute(self, sql):
            class Cur:
                @staticmethod
                def fetchone():
                    return None

            return Cur()

    @contextmanager
    def ok(timeout=None):
        yield FakeConn()

    monkeypatch.setattr(health, "connection", ok)
    assert client.get("/health/ready").status_code == 503
