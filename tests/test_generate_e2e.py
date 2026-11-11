"""Сквозной тест: загрузка → поиск → генерация → проверка ссылок → согласовано.

Настоящие Temporal (из Compose) и PostgreSQL, воркер запускается прямо в тесте.
Подменены только клиент модели и эмбеддинг запроса: модель не грузится и не вызывается.
Без базы или Temporal тесты пропускаются.
"""
import asyncio
import hashlib
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from app.core import db
from app.core.config import settings
from app.core.migrate import migrate
from app.parsers.models import Chunk
from app.rag import retrieval
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.repositories import jobs as jobs_repo
from app.schemas import Card, CritiqueReport, SourceRef, SupplierFacts
from app.services import structured
from app.temporal import activities as acts
from app.temporal.workflows import GenerateWorkflow

DIM = settings.embedding_dim
THRESHOLD = 0.6


def _basis(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


CHUNK_TEXTS = {
    "power": "Технические характеристики\nМощность: 800 Вт\nАртикул: BLD-800",
    "warranty": "Гарантийные обязательства\nГарантия 24 месяца с даты продажи.",
    "set": "Комплект поставки: блок мотора, ножка, стакан.",
}
OTHER_TEXT = "Совсем другой документ: чайник KTL-1700, мощность 2200 Вт."


@pytest.fixture(scope="module")
def env():
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            pass
    except psycopg.OperationalError:
        pytest.skip("база недоступна: поднимите make infra")

    async def ping():
        await asyncio.wait_for(Client.connect(settings.temporal_host), timeout=3)

    try:
        asyncio.run(ping())
    except Exception:
        pytest.skip("Temporal недоступен: поднимите make infra")

    migrate()
    db.init_pool()
    created = {"documents": [], "jobs": []}
    yield created
    with db.connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (created["documents"],))
        conn.execute("DELETE FROM jobs WHERE id = ANY(%s)", (created["jobs"],))
    db.close_pool()


def _indexed_document(env, name: str, texts: dict[str, str], vectors_from: int = 0) -> dict:
    """Документ в состоянии indexed с заранее заданными фрагментами и векторами."""
    digest = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    document, _ = documents_repo.create(name, digest, 1)
    env["documents"].append(document["id"])
    chunks = [
        Chunk(i, "text", text, 1 + i // 2, f"Секция {i}") for i, text in enumerate(texts.values())
    ]
    chunks_repo.replace_for_document(document["id"], name, chunks)
    rows = chunks_repo.list_without_embeddings(100, document["id"])
    vectors = [_basis(vectors_from + i) for i in range(len(rows))]
    chunks_repo.write_embeddings([(r["id"], v) for r, v in zip(rows, vectors, strict=True)])
    documents_repo.set_status(document["id"], "indexed", pages=2)
    by_key = dict(zip(texts, [r["id"] for r in rows], strict=True))
    return {"id": document["id"], "chunks": by_key}


class FakeModel:
    """Подмена клиента модели: сценарий генератора, остальные роли — по умолчанию."""

    def __init__(self, generator: list[str], facts: SupplierFacts | None = None,
                 verdicts: list[str] | None = None):
        self.generator = generator  # "valid" | "fabricated" | "unseen"
        self.facts = facts or SupplierFacts(
            product_name="Блендер МиксерПро 800",
            characteristics={"Мощность": "800 Вт", "Гарантия": "24 месяца"},
            missing_fields=["цвет"],
        )
        self.verdicts = verdicts or ["approve"] * 5
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.ids: dict[str, int] = {}  # ключ фрагмента -> chunk_id
        self.unseen_id = 0

    async def __call__(self, agent, prompt):
        self.calls.append(agent.name)
        self.prompts.append((agent.name, prompt))
        if agent.name == "Extractor":
            return self.facts
        if agent.name == "RagCritic":
            return CritiqueReport(verdict=self.verdicts.pop(0), issues=["замечание"])
        assert agent.name == "RagGenerator", agent.name
        mode = self.generator.pop(0)
        context_ids = {int(i) for i in re.findall(r'<source id="(\d+)"', prompt)}
        power, warranty = self.ids["power"], self.ids["warranty"]
        assert {power, warranty} <= context_ids  # оба фрагмента нашлись и поданы в промпт
        if mode == "fabricated":
            power = 999_999_999  # такого фрагмента нет в базе
        elif mode == "unseen":
            power = self.unseen_id  # есть в базе, но в контексте его не было
        return Card(
            title="МиксерПро 800 — блендер",
            description="Погружной блендер на 800 Вт с гарантией 24 месяца.",
            characteristics={"Мощность": "800 Вт", "Гарантия": "24 месяца"},
            benefits=["Гарантия 24 месяца"],
            sources=[
                SourceRef(field="Мощность", chunk_id=power, quote="Мощность: 800 Вт"),
                SourceRef(field="Гарантия", chunk_id=warranty, quote="Гарантия 24 месяца"),
            ],
            confidence=0.9,
        )


async def _wait(job_id: str, statuses: set[str], timeout: float = 60) -> dict:
    async with asyncio.timeout(timeout):
        while True:
            job = await asyncio.to_thread(jobs_repo.get, job_id)
            if job["status"] in statuses:
                return job
            await asyncio.sleep(0.3)


async def _run_workflow(env, document_ids, fake: FakeModel, hint="блендер", *,
                        decision: str | None = "approve") -> tuple[dict, dict | None]:
    job, _ = jobs_repo.create({"document_ids": document_ids, "hint": hint})
    env["jobs"].append(job["id"])
    client = await Client.connect(settings.temporal_host)
    queue = f"test-{uuid.uuid4().hex}"
    with ThreadPoolExecutor(max_workers=4) as executor:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[GenerateWorkflow],
            activities=[
                acts.set_status_activity,
                acts.documents_state_activity,
                acts.retrieve_activity,
                acts.rag_extract_activity,
                acts.rag_generate_activity,
                acts.verify_references_activity,
                acts.rag_critique_activity,
            ],
            activity_executor=executor,
        ):
            handle = await client.start_workflow(
                GenerateWorkflow.run,
                args=[job["id"], document_ids, hint, THRESHOLD],
                id=f"test-{job['id']}",
                task_queue=queue,
            )
            waiting = await _wait(job["id"], {"awaiting_confirmation", "needs_review", "failed"})
            final = None
            if decision and waiting["status"] != "failed":
                await handle.signal(decision)
                final = await _wait(job["id"], {"approved", "rejected"})
                await handle.result()
            return waiting, final


@pytest.fixture
def patched(monkeypatch):
    def install(fake: FakeModel, doc, other):
        fake.ids = doc["chunks"]
        fake.unseen_id = other["chunks"]["x"]
        monkeypatch.setattr(structured, "run_agent", fake)
        monkeypatch.setattr(retrieval, "embed_query", lambda text: _basis(0))
        return fake

    return install


@pytest.fixture
def docs(env):
    doc = _indexed_document(env, "blender_passport.pdf", CHUNK_TEXTS)
    other = _indexed_document(env, "kettle_manual.pdf", {"x": OTHER_TEXT}, vectors_from=10)
    return doc, other


def test_full_scenario_ends_approved_with_references_to_existing_fragments(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["valid"]), doc, other)

    waiting, final = asyncio.run(_run_workflow(env, [doc["id"]], fake))

    assert waiting["status"] == "awaiting_confirmation"
    assert final["status"] == "approved"
    card = Card.model_validate(waiting["result"])
    assert {s.chunk_id for s in card.sources} == {doc["chunks"]["power"], doc["chunks"]["warranty"]}
    for source in card.sources:  # документ, страница, секция — из базы, не со слов модели
        assert source.document == "blender_passport.pdf"
        assert source.page is not None and source.section
    assert card.missing_fields == ["цвет"]
    assert card.confidence < 1.0
    assert fake.calls == ["Extractor", "RagGenerator", "RagCritic"]
    stored_ids = chunks_repo.get_many([s.chunk_id for s in card.sources])
    assert len(stored_ids) == 2  # ссылки ведут на существующие фрагменты


def test_context_contains_only_selected_documents(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["valid"]), doc, other)
    asyncio.run(_run_workflow(env, [doc["id"]], fake))
    prompt = dict(fake.prompts[1:2])["RagGenerator"]
    assert "Мощность: 800 Вт" in prompt
    assert "KTL-1700" not in prompt  # фрагмент чужого документа в контекст не попал


def test_fabricated_reference_goes_to_rework_and_then_passes(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["fabricated", "valid"]), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [doc["id"]], fake))

    assert waiting["status"] == "awaiting_confirmation"
    assert fake.calls.count("RagGenerator") == 2  # переделка
    assert fake.calls.count("RagCritic") == 1  # критик видел только карточку с верными ссылками
    rework_prompt = [p for n, p in fake.prompts if n == "RagGenerator"][1]
    assert "999999999" in rework_prompt and "нет в базе" in rework_prompt  # текст ошибки


def test_fabricated_reference_twice_goes_to_human(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["fabricated", "fabricated", "valid"]), doc, other)

    waiting, final = asyncio.run(_run_workflow(env, [doc["id"]], fake, decision="reject"))

    assert waiting["status"] == "needs_review"
    assert "Ссылки не прошли проверку" in waiting["error"]
    assert fake.calls.count("RagGenerator") == 2  # третьей попытки нет: сразу к человеку
    assert "RagCritic" not in fake.calls  # с выдуманной ссылкой до критика не доходит
    assert final["status"] == "rejected"
    card = Card.model_validate(waiting["result"])
    assert all(s.chunk_id != 999_999_999 for s in card.sources)  # выдумка из результата убрана


def test_existing_fragment_that_model_never_saw_is_rejected(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["unseen", "unseen"]), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [doc["id"]], fake, decision=None))

    assert waiting["status"] == "needs_review"
    assert "не было в контексте" in waiting["error"]


def test_incomplete_specification_lists_real_gaps_and_low_confidence(env, docs, patched):
    doc, other = docs
    facts = SupplierFacts(
        product_name="Блендер",
        characteristics={"Мощность": "800 Вт"},
        missing_fields=["цвет", "вес", "страна-производитель"],
    )
    fake = patched(FakeModel(["valid"], facts=facts), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [doc["id"]], fake, decision=None))

    card = Card.model_validate(waiting["result"])
    assert {"цвет", "вес", "страна-производитель"} <= set(card.missing_fields)
    assert card.confidence < THRESHOLD
    assert waiting["status"] == "needs_review"  # не «готово»: карточку ждёт человек
    assert "ниже порога" in waiting["error"]


def test_critic_remarks_trigger_rework_then_approval(env, docs, patched):
    doc, other = docs
    fake = patched(FakeModel(["valid", "valid"], verdicts=["regenerate", "approve"]), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [doc["id"]], fake, decision=None))

    assert waiting["status"] == "awaiting_confirmation"
    assert fake.calls.count("RagGenerator") == 2
    assert waiting["attempts"] == 2


def test_rejected_document_is_skipped_with_note(env, docs, patched):
    doc, other = docs
    digest = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    scan, _ = documents_repo.create("boiler_scan.pdf", digest, 1)
    env["documents"].append(scan["id"])
    documents_repo.set_status(scan["id"], "failed", reason="в pdf нет текстового слоя")
    fake = patched(FakeModel(["valid"]), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [doc["id"], scan["id"]], fake, decision=None))

    assert waiting["status"] == "awaiting_confirmation"
    assert "пропущен документ" in waiting["error"]
    assert "нет текстового слоя" in waiting["error"]


def test_job_without_any_ready_document_fails_with_reason(env, patched, docs):
    doc, other = docs
    digest = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    scan, _ = documents_repo.create("boiler_scan.pdf", digest, 1)
    env["documents"].append(scan["id"])
    documents_repo.set_status(scan["id"], "failed", reason="скан без текстового слоя")
    fake = patched(FakeModel([]), doc, other)

    waiting, _ = asyncio.run(_run_workflow(env, [scan["id"]], fake, decision=None))

    assert waiting["status"] == "failed"
    assert "скан без текстового слоя" in waiting["error"]
    assert fake.calls == []  # модель не вызывалась

