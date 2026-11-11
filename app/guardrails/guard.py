"""Фасад защиты: три рубежа на пути документа к модели и результата к пользователю.

1. маскирование персональных данных (pii) — до промпта и до логов;
2. детектор инъекций (injection) — подозрительные фрагменты вон из контекста;
3. фильтр выхода (output) — контакты и следы инструкций в сгенерированной карточке.
"""
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails.injection import classify
from app.guardrails.output import check_output
from app.guardrails.pii import PiiMasker
from app.rag.context import Context, ContextItem
from app.schemas import CardDraft

log = get_logger(__name__)


class GuardReport(BaseModel):
    """Отчёт защиты: что замаскировано и что исключено. Исходных значений в нём нет."""

    masked: dict[str, int] = Field(default_factory=dict, description="вид ПД -> сколько")
    checked_fragments: int = 0
    model_checks: int = 0  # сколько раз спрашивали дешёвую модель
    excluded: list[dict] = Field(default_factory=list)
    escalated: bool = False  # документ целиком уходит к человеку
    escalation_reason: str | None = None
    output_findings: list[dict] = Field(default_factory=list)


def _merge_counts(*counts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for part in counts:
        for kind, number in part.items():
            merged[kind] = merged.get(kind, 0) + number
    return merged


async def guard_context(context: Context) -> tuple[Context, GuardReport]:
    """Рубежи 1 и 2 для контекста: маскирование и детектор инъекций.

    Маскирование идемпотентно: уже замаскированный текст не меняется. Подозрительные
    фрагменты выбрасываются; если их не меньше порога из настроек или не осталось ни одного —
    документ целиком уходит к человеку (`escalated`) с причиной.
    """
    masker = PiiMasker()
    kept: list[ContextItem] = []
    excluded: list[dict] = []
    model_checks = 0
    for item in context.items:
        masked = ContextItem(
            chunk_id=item.chunk_id,
            document=item.document,
            page=item.page,
            section=masker.mask(item.section),
            text=masker.mask(item.text),
        )
        result = await classify(masked.text)
        model_checks += result.model_called
        if not result.suspicious:
            kept.append(masked)
            continue
        excluded.append(
            {
                "chunk_id": item.chunk_id,
                "document": item.document,
                "page": item.page,
                "section": masked.section,
                "level": result.level,
                "rules": result.rules,
                "reason": result.reason,
            }
        )
        log.warning(
            "fragment_excluded", chunk_id=item.chunk_id, level=result.level, rules=result.rules
        )

    escalation = None
    if len(excluded) >= settings.injection_max_suspicious:
        escalation = (
            f"подозрительных фрагментов {len(excluded)} "
            f"(порог {settings.injection_max_suspicious}): документ отправлен человеку"
        )
    elif not kept:
        escalation = "все найденные фрагменты исключены как подозрительные"

    report = GuardReport(
        masked=_merge_counts(context.pii_masked, masker.counts),
        checked_fragments=len(context.items),
        model_checks=model_checks,
        excluded=excluded,
        escalated=escalation is not None,
        escalation_reason=escalation,
    )
    guarded = Context(
        items=kept,
        duplicates_removed=context.duplicates_removed,
        omitted=context.omitted,
        pii_masked=report.masked,
    )
    return guarded, report


def guard_output(
    card: CardDraft, excluded_texts: list[str] | None = None
) -> tuple[CardDraft, GuardReport]:
    """Рубеж 3: контакты в карточке заменяются метками, находки попадают в отчёт."""
    checked = check_output(card, excluded_texts)
    return checked.card, GuardReport(output_findings=checked.findings)
