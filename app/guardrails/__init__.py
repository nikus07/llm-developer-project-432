"""Защита документа как недоверенных данных: три рубежа на пути к модели и обратно.

1. персональные данные маскируются до промпта и до логов (guardrails.pii);
2. детектор инъекций: правила по тексту, затем вопрос дешёвой модели по помеченным
   фрагментам; подозрительное выбрасывается (guardrails.injection);
3. фильтр выхода: контакты и следы инструкций в сгенерированной карточке (guardrails.output).
"""
from pydantic import BaseModel, Field, computed_field

from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails.injection import (
    InjectionVerdict,
    detect_injection_llm,
    detect_injection_regex,
)
from app.guardrails.output import check_output, check_output_json
from app.guardrails.pii import PiiMasker
from app.schemas import CardDraft

log = get_logger(__name__)

__all__ = ["GuardReport", "guard_context", "guard_output"]


class GuardReport(BaseModel):
    """Отчёт защиты: что замаскировано и что исключено. Исходных значений в нём нет."""

    masked_pii: int = 0  # сколько персональных данных замаскировано
    masked: dict[str, int] = Field(default_factory=dict, description="вид ПД -> сколько")
    suspicious_chunks: list[str] = Field(default_factory=list)  # id исключённых фрагментов
    excluded: list[dict] = Field(default_factory=list)  # подробности: уровень, правила, причина
    checked_chunks: int = 0
    model_checks: int = 0  # сколько раз спрашивали дешёвую модель
    output_issues: list[str] = Field(default_factory=list)  # замечания фильтра выхода
    # замаскированные тексты исключённого: нужны фильтру выхода, в отчёт не попадают
    excluded_texts: list[str] = Field(default_factory=list, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_review(self) -> bool:
        """Много подозрительных фрагментов или не осталось ни одного: документ — человеку."""
        count = len(self.suspicious_chunks)
        return count >= settings.suspicious_chunk_limit or (
            self.checked_chunks > 0 and count == self.checked_chunks
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def escalation_reason(self) -> str | None:
        if not self.needs_review:
            return None
        count = len(self.suspicious_chunks)
        if count >= settings.suspicious_chunk_limit:
            return (
                f"подозрительных фрагментов {count} "
                f"(порог {settings.suspicious_chunk_limit}): документ отправлен человеку"
            )
        return "все найденные фрагменты исключены как подозрительные"


async def guard_context(chunks: list[dict]) -> tuple[list[dict], GuardReport]:
    """Рубежи 1 и 2 для найденных фрагментов: маскирование и детектор инъекций.

    Фрагмент — словарь с ключами chunk_id и text (остальные ключи сохраняются). Маскирование
    идемпотентно. Подозрительные фрагменты выбрасываются; если детектор недоступен —
    фрагмент считается подозрительным (защита, которая при отказе открывает дверь, —
    не защита). Платная модель спрашивается только по фрагментам, помеченным правилами.
    """
    masker = PiiMasker()
    safe: list[dict] = []
    excluded: list[dict] = []
    excluded_texts: list[str] = []
    model_checks = 0
    for chunk in chunks:
        text = masker.mask(chunk["text"])
        verdict = detect_injection_regex(text)
        if verdict.suspicious:
            model_checks += 1
            try:
                verdict = await detect_injection_llm(text)
            except Exception as exc:
                log.warning("injection_detector_unavailable", error=repr(exc)[:200])
                verdict = InjectionVerdict(
                    True,
                    "детектор недоступен: фрагмент считается подозрительным",
                    verdict.rules,
                    "fail_closed",
                )
        if verdict.suspicious:
            log.warning("fragment_excluded", chunk_id=chunk["chunk_id"], level=verdict.level)
            metadata = chunk.get("metadata") or {}
            excluded.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk.get("filename"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "level": verdict.level,
                    "rules": verdict.rules,
                    "reason": verdict.reason,
                }
            )
            excluded_texts.append(text)
            continue
        safe.append({**chunk, "text": text})

    report = GuardReport(
        masked_pii=sum(masker.counts.values()),
        masked=dict(masker.counts),
        suspicious_chunks=[e["chunk_id"] for e in excluded],
        excluded=excluded,
        checked_chunks=len(chunks),
        model_checks=model_checks,
        excluded_texts=excluded_texts,
    )
    return safe, report


def guard_output(card, excluded_texts: list[str] | None = None):
    """Рубеж 3: контакты в карточке заменяются метками, находки возвращаются замечаниями.

    Карточка — строка JSON (получаем (очищенная строка, замечания)) или CardDraft
    (получаем (CardDraft, замечания)).
    """
    if isinstance(card, CardDraft):
        checked = check_output(card, excluded_texts)
        return checked.card, [f"{f['kind']} ({f['field']})" for f in checked.findings]
    return check_output_json(card, excluded_texts)
