"""Стадии пайплайна по тексту поставщика. Один код для FastAPI и Temporal."""
from pydantic import ValidationError

from app.agents.prompts import critic_agent, extractor_agent
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas import CardDraft, CritiqueReport, SupplierFacts
from app.services import structured
from app.services.json_utils import LLMJsonError
from app.services.structured import confidence_gate, parse_model

log = get_logger(__name__)


def apply_facts_gaps(card: CardDraft, facts: SupplierFacts) -> CardDraft:
    """Отказ вместо догадки: недостающее из фактов попадает в карточку кодом, не промптом.

    - всё, чего нет в фактах, идёт в missing_fields и убирается из характеристик
      (значение, которого нет в документах, — выдумка);
    - уверенность ограничена сверху: чем больше пробелов, тем ниже потолок.
    """
    missing = list(dict.fromkeys([*card.missing_fields, *facts.missing_fields]))
    characteristics = {k: v for k, v in card.characteristics.items() if k not in missing}
    confidence = card.confidence
    if not facts.characteristics:
        confidence = 0.0
    elif missing:
        confidence = min(confidence, max(0.0, 0.9 - 0.2 * len(missing)))
    return card.model_copy(
        update={
            "missing_fields": missing,
            "characteristics": characteristics,
            "confidence": confidence,
        }
    )


async def _ask(agent, prompt: str, model_cls):
    """Вызов агента и разбор ответа; при невалидном ответе один повтор с текстом ошибки."""
    out = await run_agent(agent, prompt)
    try:
        return parse_model(out, model_cls)
    except (LLMJsonError, ValidationError) as exc:
        log.warning("pipeline_invalid_answer", model=model_cls.__name__, error=str(exc)[:200])
        retry = (
            f"{prompt}\n\nТвой предыдущий ответ:\n{out}\n\n"
            f"Он не прошёл проверку: {exc}\nВерни исправленный JSON целиком, без пояснений."
        )
        return parse_model(await run_agent(agent, retry), model_cls)


async def extract(supplier_text: str) -> SupplierFacts:
    log.info("step_extract_start")
    return await _ask(extractor_agent(), supplier_text, SupplierFacts)


async def generate(facts: SupplierFacts, feedback: list[str] | None = None) -> CardDraft:
    prompt = "Факты о товаре:\n" + facts.model_dump_json(indent=2)
    if feedback:
        prompt += "\n\nЗамечания критика, учти:\n" + "\n".join(f"- {i}" for i in feedback)
    log.info("step_generate_start", feedback=bool(feedback))
    card = await structured.validate_or_retry(prompt)
    return apply_facts_gaps(card, facts)


async def critique(facts: SupplierFacts, draft: CardDraft) -> CritiqueReport:
    prompt = (
        "Факты о товаре:\n"
        + facts.model_dump_json(indent=2)
        + "\n\nЧерновик карточки:\n"
        + draft.model_dump_json(indent=2)
    )
    log.info("step_critique_start")
    return await _ask(critic_agent(), prompt, CritiqueReport)


async def run_pipeline(supplier_text: str, max_attempts: int = 3):
    """Текст поставщика → карточка. Возвращает (черновик, попыток, вердикт, статус).

    Вердикт — approved | rejected (проверяющая роль), статус — done | needs_review | rejected:
    одобренная карточка с низкой уверенностью уходит человеку, а не в «готово».
    """
    facts = await extract(supplier_text)  # факты не зависят от замечаний критика
    feedback: list[str] = []
    attempts = 0
    draft = None
    verdict = "rejected"
    while attempts < max_attempts:
        attempts += 1
        draft = await generate(facts, feedback or None)
        report = await critique(facts, draft)
        if report.verdict == "approve":
            verdict = "approved"
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=report.issues)
    status = confidence_gate(draft) if verdict == "approved" else "rejected"
    return draft, attempts, verdict, status
