"""Шаги процесса. Стадии — те же функции из services/pipeline.py, что и в синхронном варианте."""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from openai import APIStatusError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from app.guardrails import GuardReport
from app.llm.client import _retryable
from app.repositories import documents as documents_repo
from app.repositories import jobs as jobs_repo
from app.schemas import CardDraft, SupplierFacts
from app.services import documents as documents_service
from app.services import pipeline, rag_pipeline


def _to_application_error(exc: Exception) -> Exception:
    """Ошибка в запросе к модели повтором шага не лечится — не повторяем."""
    if isinstance(exc, APIStatusError) and not _retryable(exc):
        return ApplicationError(repr(exc), non_retryable=True)
    return exc


@asynccontextmanager
async def _heartbeating(interval_s: float = 10.0) -> AsyncIterator[None]:
    """Сердцебиение на время долгого вызова модели.

    Без него движок узнал бы о смерти воркера только по таймауту всего шага (минуты).
    """

    async def beat() -> None:
        while True:
            activity.heartbeat()
            await asyncio.sleep(interval_s)

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@activity.defn
async def extract_activity(supplier_text: str) -> dict:
    try:
        async with _heartbeating():
            facts = await pipeline.extract(supplier_text)
    except Exception as exc:
        raise _to_application_error(exc) from exc
    return facts.model_dump()


@activity.defn
async def generate_activity(args: dict) -> dict:
    facts = SupplierFacts.model_validate(args["facts"])
    try:
        async with _heartbeating():
            draft = await pipeline.generate(facts, args.get("feedback") or None)
    except Exception as exc:
        raise _to_application_error(exc) from exc
    return draft.model_dump()


@activity.defn
async def critique_activity(args: dict) -> dict:
    facts = SupplierFacts.model_validate(args["facts"])
    draft = CardDraft.model_validate(args["draft"])
    try:
        async with _heartbeating():
            report = await pipeline.critique(facts, draft)
    except Exception as exc:
        raise _to_application_error(exc) from exc
    return report.model_dump()


@activity.defn
def set_status_activity(args: dict) -> None:
    """Статус для клиента пишется отдельным шагом, а не изнутри пайплайна.

    Обычная функция (не async): обращение к базе блокирует, поэтому движок
    исполняет её в пуле потоков воркера.
    """
    jobs_repo.update_status(
        args["job_id"],
        args["status"],
        attempts=args.get("attempts"),
        result=args.get("result"),
        error=args.get("error"),
        report=args.get("report"),
    )


@activity.defn
def process_document_activity(document_id: str) -> dict:
    """Блокирующий шаг (диск, разбор файла, база): исполняется в потоке воркера."""
    return documents_service.process_document(document_id)


@activity.defn
def mark_document_failed_activity(args: dict) -> None:
    documents_repo.set_status(args["document_id"], "failed", reason=args["reason"])


@activity.defn
def index_document_activity(document_id: str) -> dict:
    """Блокирующий шаг: эмбеддинги считаются на процессоре, модель грузится один раз."""
    return documents_service.index_document(document_id)


# --- генерация по документам (сквозной процесс) ---


@activity.defn
def documents_state_activity(document_ids: list[str]) -> dict:
    """Что готово, что ещё разбирается, что отвергнуто (короткое обращение к базе)."""
    return rag_pipeline.documents_state(document_ids)


@activity.defn
async def retrieve_activity(args: dict) -> dict:
    """Поиск, маскирование ПД и детектор инъекций (дешёвая модель по помеченным фрагментам)."""
    query = rag_pipeline.base_query(args["document_ids"], args.get("hint"))
    async with _heartbeating():
        context, guard = await rag_pipeline.retrieve_context_async(query, args["document_ids"])
    return {
        "query": query,
        "context": context,
        "guard": guard.model_dump(),
        "excluded_texts": guard.excluded_texts,
        "needs_review": guard.needs_review,
        "escalation_reason": guard.escalation_reason,
    }


@activity.defn
async def rag_card_activity(args: dict) -> dict:
    """Генерация карточки по контексту: проверка ссылок, критик, фильтр выхода."""
    guard = GuardReport.model_validate(
        {**args["guard"], "excluded_texts": args.get("excluded_texts") or []}
    )
    try:
        async with _heartbeating():
            card, attempts, verdict, status, guard = await rag_pipeline.generate_from_context(
                args["query"], args["context"], guard
            )
    except rag_pipeline.SourceValidationError as exc:
        # выдуманные ссылки не лечатся повтором шага: карточку получит человек
        raise ApplicationError(str(exc), non_retryable=True) from exc
    except Exception as exc:
        raise _to_application_error(exc) from exc
    return {
        "card": card.model_dump(),
        "attempts": attempts,
        "verdict": verdict,
        "status": status,
        "guard": guard.model_dump(),
    }
