"""Загрузка и обработка документов. База нужна не везде: проверки запроса идут без неё."""
import hashlib
import shutil
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.core import db
from app.core.config import settings
from app.core.migrate import migrate
from app.main import app
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.routers import documents as documents_router
from app.services import documents as documents_service

DATA = Path(__file__).resolve().parent.parent / "data"
client = TestClient(app)
_created_ids: list[str] = []


# --- проверка запроса: до базы дело не доходит ---


def test_unsupported_extension_is_rejected():
    resp = client.post("/documents", files={"file": ("virus.exe", b"MZ")})
    assert resp.status_code == 415


def test_empty_file_is_rejected():
    resp = client.post("/documents", files={"file": ("a.pdf", b"")})
    assert resp.status_code == 400


def test_too_big_file_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    resp = client.post("/documents", files={"file": ("a.pdf", b"x")})
    assert resp.status_code == 413


# --- живая база ---


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
    with db.connection() as conn:  # убираем за собой документы, созданные тестами
        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (_created_ids,))
    db.close_pool()


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    return tmp_path


def _register(name: str, content: bytes) -> dict:
    digest = hashlib.sha256(content + uuid.uuid4().bytes).hexdigest()
    document, created = documents_repo.create(name, digest, len(content))
    assert created
    _created_ids.append(document["id"])
    documents_service.save_upload(document["id"], name, content)
    return document


@pytest.mark.parametrize(
    "name", ["blender_passport.pdf", "kettle_manual.pdf", "blender_kp.docx", "kettle_spec.xlsx"]
)
def test_supplied_files_are_parsed(live_db, uploads, name):
    document = _register(name, (DATA / name).read_bytes())
    assert document["status"] == "new"

    result = documents_service.process_document(document["id"])

    assert result["status"] == "parsed"
    stored = documents_repo.get(document["id"])
    assert stored["status"] == "parsed"
    assert stored["chunks"] == result["chunks"] > 0
    for chunk in chunks_repo.list_for_document(document["id"]):
        assert chunk["document"] == name
        assert chunk["page"] >= 1
        assert chunk["section"]


def test_scanned_pdf_gets_rejection_with_reason(live_db, uploads):
    document = _register("boiler_scan.pdf", (DATA / "boiler_scan.pdf").read_bytes())

    result = documents_service.process_document(document["id"])

    assert result["status"] == "failed"
    stored = documents_repo.get(document["id"])
    assert stored["status"] == "failed"
    assert "нет текстового слоя" in stored["reason"]
    assert stored["chunks"] == 0


def test_spec_chunks_read_as_article_parameter_value(live_db, uploads):
    document = _register("kettle_spec.xlsx", (DATA / "kettle_spec.xlsx").read_bytes())
    documents_service.process_document(document["id"])

    rows = chunks_repo.list_for_document(document["id"])
    joined = "\n".join(r["text"] for r in rows)
    assert "KTL-1700: бренд КеттлПро" in joined
    assert all(r["brand"] == "КеттлПро" for r in rows)


def test_same_file_does_not_create_duplicate(live_db):
    content = b"%PDF-1.4 same file " + hashlib.sha256(b"dup").digest()
    digest = hashlib.sha256(content).hexdigest()
    with db.connection() as conn:
        conn.execute("DELETE FROM documents WHERE content_hash = %s", (digest,))
    first, created_first = documents_repo.create("a.pdf", digest, len(content))
    _created_ids.append(first["id"])
    second, created_second = documents_repo.create("copy-of-a.pdf", digest, len(content))
    assert created_first is True
    assert created_second is False
    assert second["id"] == first["id"]


def test_reprocessing_replaces_chunks_without_duplicates(live_db, uploads):
    document = _register("kettle_spec.xlsx", (DATA / "kettle_spec.xlsx").read_bytes())
    first = documents_service.process_document(document["id"])
    second = documents_service.process_document(document["id"])
    assert first["chunks"] == second["chunks"]
    assert documents_repo.get(document["id"])["chunks"] == first["chunks"]


def test_upload_endpoint_returns_202_and_reuses_duplicate(live_db, uploads, monkeypatch):
    started: list[str] = []

    async def fake_start(document_id: str) -> None:
        started.append(document_id)

    monkeypatch.setattr(documents_router, "start_document_workflow", fake_start)
    content = (DATA / "kettle_spec.xlsx").read_bytes() + b"\x00endpoint-test"
    with db.connection() as conn:
        digest = hashlib.sha256(content).hexdigest()
        conn.execute("DELETE FROM documents WHERE content_hash = %s", (digest,))

    first = client.post("/documents", files={"file": ("kettle_spec.xlsx", content)})
    second = client.post("/documents", files={"file": ("again.xlsx", content)})

    _created_ids.append(first.json()["id"])
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert started == [first.json()["id"]]  # разбор запущен один раз
    assert client.get(f"/documents/{first.json()['id']}").json()["status"] == "new"
    assert client.get("/documents/нет-такого").status_code == 404
    shutil.rmtree(uploads, ignore_errors=True)
