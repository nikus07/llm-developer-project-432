"""Генерация карточки по документам: поиск → контекст → карточка → проверка ссылок.

Каждая функция — один шаг сквозного процесса; Temporal вызывает их как отдельные шаги,
а тесты — напрямую. Три рубежа против выдумывания: поиск (модель видит только найденное),
проверка ссылок (здесь, кодом) и проверяющая роль (critique).
"""
from pathlib import PurePath

from app.agents.prompts import rag_critic_agent, rag_generator_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails.injection import classify
from app.guardrails.output import check_output
from app.guardrails.pii import PiiMasker
from app.rag.context import Context, build_context
from app.rag.retrieval import search
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas import Card, CritiqueReport, SupplierFacts
from app.services import pipeline
from app.services.structured import call_structured

log = get_logger(__name__)

_NOT_READY = ("new", "parsing", "parsed")


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


def _base_query(document_ids: list[str], hint: str | None) -> str:
    if hint and hint.strip():
        return hint.strip()
    names = []
    for document_id in document_ids:
        document = documents_repo.get(document_id)
        if document:
            names.append(PurePath(document["filename"]).stem.replace("_", " "))
    return " ".join(names)


def retrieve_context(document_ids: list[str], hint: str | None) -> Context:
    """Гибридный поиск внутри выбранных документов по подсказке или заголовку документа."""
    base = _base_query(document_ids, hint)
    queries = [base, *(f"{base} {aspect}" for aspect in settings.generate_aspects)]
    hits: list[dict] = []
    seen: set[int] = set()
    for query in queries:
        found = search(
            query,
            mode="hybrid",
            limit=settings.generate_top_k,
            min_score=settings.generate_min_score,
            filters={"document_ids": document_ids},
        )
        for hit in found:
            if hit["id"] not in seen:
                seen.add(hit["id"])
                hits.append(hit)
    # Первый рубеж: персональные данные маскируются на фрагменте при сборке контекста, до
    # промпта, до детектора инъекций и до любой записи в лог. В базе остаётся исходный текст.
    masker = PiiMasker()
    for hit in hits:
        hit["text"] = masker.mask(hit["text"])
        hit["section"] = masker.mask(hit["section"])
    context = build_context(hits)
    context.pii_masked = dict(masker.counts)
    log.info(
        "context_built",
        fragments=len(context.items),
        duplicates_removed=context.duplicates_removed,
        omitted=context.omitted,
        pii_masked=context.pii_masked,
    )
    return context


async def extract_facts(context: Context) -> SupplierFacts:
    """Факты и недостающие поля — по тексту найденных источников, а не всего документа."""
    return await pipeline.extract(context.text)


def _generation_prompt(
    facts: SupplierFacts, context: Context, hint: str | None, feedback: list[str] | None
) -> str:
    parts = []
    if hint:
        parts.append(f"Подсказка о товаре: {hint}")
    parts.append("Факты (JSON):\n" + facts.model_dump_json(indent=2))
    parts.append("Источники:\n" + context.text)
    if feedback:
        parts.append("Замечания проверки, исправь:\n" + "\n".join(f"- {i}" for i in feedback))
    return "\n\n".join(parts)


async def generate_card(
    facts: SupplierFacts,
    context: Context,
    hint: str | None = None,
    feedback: list[str] | None = None,
) -> Card:
    log.info("rag_generate_start", fragments=len(context.items), feedback=bool(feedback))
    prompt = _generation_prompt(facts, context, hint, feedback)
    card = await call_structured(rag_generator_agent(), prompt, Card, field_repair=True)
    return pipeline.apply_facts_gaps(card, facts)


def find_reference_issues(
    card: Card, context_ids: set[int], existing_ids: set[int]
) -> list[str]:
    """Проверка ссылок. Каждая должна пройти два условия:

    1. такой фрагмент есть в базе;
    2. он был в поданном контексте.

    Второе важнее: сослаться на существующий фрагмент, которого модель не видела, —
    выдумывание с правдоподобным видом.
    """
    issues: list[str] = []
    for source in card.sources:
        if source.chunk_id is None:
            issues.append(f"у ссылки для «{source.field}» нет chunk_id")
        elif source.chunk_id not in existing_ids:
            issues.append(
                f"ссылка для «{source.field}»: фрагмента {source.chunk_id} нет в базе"
            )
        elif source.chunk_id not in context_ids:
            issues.append(
                f"ссылка для «{source.field}»: фрагмента {source.chunk_id} не было в контексте, "
                "ссылайся только на переданные источники"
            )

    cited = {s.field.casefold() for s in card.sources}
    uncited = [name for name in card.characteristics if name.casefold() not in cited]
    if uncited:
        issues.append(f"нет ссылки на источник для характеристик: {', '.join(uncited)}")
    return issues


def verify_references(card: Card, context: Context) -> tuple[Card, list[str]]:
    """Проверяет ссылки и оставляет в карточке только верные, дополнив их метаданными.

    Возвращает (карточка, нарушения). Документ, страница и секция берутся из базы,
    а не из слов модели.
    """
    ids = [s.chunk_id for s in card.sources if s.chunk_id is not None]
    stored = chunks_repo.get_many(ids)
    issues = find_reference_issues(card, context.chunk_ids, set(stored))

    valid = []
    for source in card.sources:
        row = stored.get(source.chunk_id) if source.chunk_id is not None else None
        if row is None or source.chunk_id not in context.chunk_ids:
            continue
        valid.append(
            source.model_copy(
                update={"document": row["document"], "page": row["page"], "section": row["section"]}
            )
        )
    return card.model_copy(update={"sources": valid}), issues


async def critique_card(facts: SupplierFacts, context: Context, card: Card) -> CritiqueReport:
    """Проверяющая роль: каждое утверждение карточки сверяется с заявленным источником."""
    prompt = (
        "Источники:\n"
        + context.text
        + "\n\nКарточка:\n"
        + card.model_dump_json(indent=2)
        + "\n\nФакты (JSON):\n"
        + facts.model_dump_json(indent=2)
    )
    log.info("rag_critique_start")
    return await call_structured(rag_critic_agent(), prompt, CritiqueReport)


async def screen_context(context: Context) -> tuple[Context, dict, list[str]]:
    """Второй рубеж: детектор инъекций. Подозрительные фрагменты — вон из контекста.

    Возвращает (очищенный контекст, отчёт, тексты исключённых для фильтра выхода). Если
    подозрительных не меньше порога из настроек или не осталось ни одного фрагмента —
    документ целиком уходит к человеку с причиной. В отчёте нет исходных значений.
    """
    kept = []
    excluded: list[dict] = []
    excluded_texts: list[str] = []
    model_checks = 0
    for item in context.items:
        result = await classify(item.text)
        model_checks += result.model_called
        if not result.suspicious:
            kept.append(item)
            continue
        excluded_texts.append(item.text)
        excluded.append(
            {
                "chunk_id": item.chunk_id,
                "document": item.document,
                "page": item.page,
                "section": item.section,
                "level": result.level,
                "rules": result.rules,
                "reason": result.reason,
            }
        )
        log.warning("fragment_excluded", chunk_id=item.chunk_id, level=result.level,
                    rules=result.rules)

    escalation = None
    if len(excluded) >= settings.injection_max_suspicious:
        escalation = (
            f"подозрительных фрагментов {len(excluded)} "
            f"(порог {settings.injection_max_suspicious}): документ отправлен человеку"
        )
    elif not kept:
        escalation = "все найденные фрагменты исключены как подозрительные"

    report = {
        "masked": context.pii_masked,
        "checked_fragments": len(context.items),
        "model_checks": model_checks,
        "excluded": excluded,
        "escalated": escalation is not None,
        "escalation_reason": escalation,
    }
    screened = Context(
        items=kept,
        duplicates_removed=context.duplicates_removed,
        omitted=context.omitted,
        pii_masked=context.pii_masked,
    )
    return screened, report, excluded_texts


def filter_output(card: Card, excluded_texts: list[str]) -> tuple[Card, list[dict]]:
    """Третий рубеж: контакты и следы инструкций в сгенерированной карточке."""
    checked = check_output(card, excluded_texts)
    return checked.card, checked.findings
