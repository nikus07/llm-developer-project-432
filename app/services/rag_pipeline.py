"""Генерация карточки по документам: поиск → защита → контекст → карточка → проверка ссылок.

Три рубежа против выдумывания: поиск (модель видит только найденное), проверка ссылок
(здесь, кодом) и проверяющая роль (Critic). Один код для FastAPI-вызова, Temporal и тестов.
"""
import asyncio
from pathlib import PurePath

from pydantic import ValidationError

from app.agents.prompts import rag_critic_agent, rag_generator_agent
from app.core.db import connection
from app.core.logging import get_logger
from app.guardrails import GuardReport, guard_context, guard_output
from app.llm.client import run_agent
from app.rag.context import build_context, context_chunk_ids
from app.rag.retrieval import retrieve_chunks
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas import CardDraft, CritiqueReport
from app.services.json_utils import LLMJsonError
from app.services.structured import confidence_gate, parse_model

log = get_logger(__name__)

MAX_ATTEMPTS = 3


class SourceValidationError(Exception):
    """Карточка ссылается на фрагменты, которых нет в базе или не было в контексте."""


def documents_state(document_ids: list[str]) -> dict:
    """Готовность документов: что ещё разбирается, что готово, что отвергнуто."""
    ready: list[str] = []
    skipped: list[dict] = []
    waiting: list[str] = []
    parsing = False
    for document_id in document_ids:
        document = documents_repo.get(document_id)
        if document is None:
            skipped.append({"id": document_id, "reason": "документ не найден"})
        elif document["status"] == "failed":
            skipped.append({"id": document_id, "reason": document["reason"]})
        elif document["status"] == "indexed":
            ready.append(document_id)
        else:
            waiting.append(document_id)
            parsing = parsing or document["status"] in ("new", "parsing")
    stage = "parsing" if parsing else "indexing"
    return {"ready": ready, "skipped": skipped, "waiting": waiting, "stage": stage}


def base_query(document_ids: list[str], hint: str | None) -> str:
    """Запрос для поиска: подсказка пользователя или имена документов."""
    if hint and hint.strip():
        return hint.strip()
    names = []
    for document_id in document_ids:
        document = documents_repo.get(document_id)
        if document:
            names.append(PurePath(document["filename"]).stem.replace("_", " "))
    return " ".join(names)


async def retrieve_context_async(
    query: str, doc_ids: list[str] | None = None
) -> tuple[str, GuardReport]:
    """Поиск → защита (маскирование ПД, детектор инъекций) → контекст-строка с заголовками."""
    hits = await asyncio.to_thread(retrieve_chunks, query, doc_ids)
    safe, report = await guard_context(hits)
    context = build_context(safe)
    log.info(
        "context_built",
        found=len(hits),
        kept=len(safe),
        excluded=len(report.suspicious_chunks),
        masked=report.masked,
    )
    return context, report


def _generation_prompt(query: str, context: str, feedback: list[str] | None) -> str:
    parts = [f"Запрос: {query}", "Источники:\n" + context]
    if feedback:
        parts.append("Замечания проверки, исправь:\n" + "\n".join(f"- {i}" for i in feedback))
    return "\n\n".join(parts)


async def _ask(agent, prompt: str, model_cls):
    """Вызов агента и разбор ответа; при невалидном ответе один повтор с текстом ошибки."""
    out = await run_agent(agent, prompt)
    try:
        return parse_model(out, model_cls)
    except (LLMJsonError, ValidationError) as exc:
        log.warning("rag_invalid_answer", model=model_cls.__name__, error=str(exc)[:200])
        retry = (
            f"{prompt}\n\nТвой предыдущий ответ:\n{out}\n\n"
            f"Он не прошёл проверку: {exc}\nВерни исправленный JSON целиком, без пояснений."
        )
        return parse_model(await run_agent(agent, retry), model_cls)


def _with_gaps(card: CardDraft) -> CardDraft:
    """Недостающее не выдумывается: поле из missing_fields нет в характеристиках, а
    уверенность ограничена сверху числом пробелов."""
    missing = list(dict.fromkeys(card.missing_fields))
    characteristics = {k: v for k, v in card.characteristics.items() if k not in missing}
    confidence = card.confidence
    if missing:
        confidence = min(confidence, max(0.0, 0.9 - 0.2 * len(missing)))
    return card.model_copy(
        update={
            "missing_fields": missing,
            "characteristics": characteristics,
            "confidence": confidence,
        }
    )


def find_reference_issues(
    card: CardDraft, context_ids: set[str], existing_ids: set[str]
) -> list[str]:
    """Проверка ссылок. Каждая должна пройти два условия:

    1. такой фрагмент есть в базе;
    2. он был в поданном контексте.

    Второе важнее: сослаться на существующий фрагмент, которого модель не видела, —
    выдумывание с правдоподобным видом.
    """
    issues: list[str] = []
    for source in card.sources:
        label = source.field or source.chunk_id
        if source.chunk_id not in existing_ids:
            issues.append(f"ссылка для «{label}»: фрагмента {source.chunk_id} нет в базе")
        elif source.chunk_id not in context_ids:
            issues.append(
                f"ссылка для «{label}»: фрагмента {source.chunk_id} не было в контексте, "
                "ссылайся только на переданные источники"
            )
    if card.characteristics and not card.sources:
        issues.append("у карточки с характеристиками нет ни одной ссылки на источник")
    return issues


def verify_references(card: CardDraft, context: str) -> tuple[CardDraft, list[str]]:
    """Проверяет ссылки и оставляет только верные, дополнив документом, страницей, секцией.

    Возвращает (карточка, нарушения). Подробности берутся из базы, а не из слов модели.
    """
    in_context = context_chunk_ids(context)
    with connection() as conn:
        stored = chunks_repo.get_chunks_by_ids(conn, [s.chunk_id for s in card.sources])
    issues = find_reference_issues(card, in_context, set(stored))

    valid = []
    for source in card.sources:
        row = stored.get(source.chunk_id)
        if row is None or source.chunk_id not in in_context:
            continue
        meta = row["metadata"] or {}
        valid.append(
            source.model_copy(
                update={
                    "doc_id": row["doc_id"],
                    "document": row["filename"],
                    "page": meta.get("page", source.page),
                    "section": meta.get("section"),
                }
            )
        )
    return card.model_copy(update={"sources": valid}), issues


async def critique_card(card: CardDraft, context: str) -> CritiqueReport:
    """Проверяющая роль: каждое утверждение карточки сверяется с заявленным источником."""
    prompt = "Источники:\n" + context + "\n\nКарточка:\n" + card.model_dump_json(indent=2)
    log.info("rag_critique_start")
    return await _ask(rag_critic_agent(), prompt, CritiqueReport)


async def generate_from_context(
    query: str, context: str, guard: GuardReport, max_attempts: int = 3
) -> tuple[CardDraft, int, str, str, GuardReport]:
    """Карточка по готовому контексту: (карточка, попыток, вердикт, статус, отчёт защиты).

    Невалидная ссылка — замечание на следующую попытку; если ссылки не исправились за все
    попытки, SourceValidationError (выдуманный источник нельзя отдавать клиенту).
    """
    feedback: list[str] = []
    attempts = 0
    card: CardDraft | None = None
    verdict = "rejected"
    reference_issues: list[str] = []
    while attempts < max_attempts:
        attempts += 1
        log.info("rag_generate_start", attempt=attempts, feedback=bool(feedback))
        draft = await _ask(
            rag_generator_agent(), _generation_prompt(query, context, feedback), CardDraft
        )
        card, reference_issues = await asyncio.to_thread(
            verify_references, _with_gaps(draft), context
        )
        if reference_issues:
            feedback = reference_issues
            continue
        report = await critique_card(card, context)
        if report.verdict == "approve":
            verdict = "approved"
            break
        feedback = report.issues
        log.info("rag_regenerate", attempt=attempts, issues=report.issues)
    if reference_issues:
        raise SourceValidationError("; ".join(reference_issues))

    card, issues = guard_output(card, guard.excluded_texts)
    guard.output_issues = issues
    status = confidence_gate(card) if verdict == "approved" else "rejected"
    # контакты в карточке заменены метками, но сам факт находки — повод для человека
    if status == "done" and (guard.needs_review or issues):
        status = "needs_review"
    return card, attempts, verdict, status, guard


async def run_rag_pipeline(
    query: str, doc_ids: list[str] | None = None, max_attempts: int | None = None
) -> tuple[CardDraft, int, str, str, GuardReport]:
    """Запрос и документы → карточка со ссылками: поиск, защита, генерация, проверка."""
    context, guard = await retrieve_context_async(query, doc_ids)
    if not context.strip():
        raise SourceValidationError("в выбранных документах ничего не найдено")
    return await generate_from_context(
        query, context, guard, max_attempts or MAX_ATTEMPTS
    )
