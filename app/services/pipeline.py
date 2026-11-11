"""Стадии пайплайна. Один код для FastAPI и Temporal."""
from app.agents.prompts import critic_agent, extractor_agent, generator_agent
from app.core.logging import get_logger
from app.schemas import CardDraft, CritiqueReport, SupplierFacts
from app.services.structured import call_structured

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


async def extract(supplier_text: str) -> SupplierFacts:
    log.info("step_extract_start")
    return await call_structured(extractor_agent(), supplier_text, SupplierFacts)


async def generate(facts: SupplierFacts, feedback: list[str] | None = None) -> CardDraft:
    prompt = "Факты о товаре:\n" + facts.model_dump_json(indent=2)
    if feedback:
        prompt += "\n\nЗамечания критика, учти:\n" + "\n".join(f"- {i}" for i in feedback)
    log.info("step_generate_start", feedback=bool(feedback))
    card = await call_structured(generator_agent(), prompt, CardDraft, field_repair=True)
    return apply_facts_gaps(card, facts)


async def critique(facts: SupplierFacts, draft: CardDraft) -> CritiqueReport:
    prompt = (
        "Факты о товаре:\n"
        + facts.model_dump_json(indent=2)
        + "\n\nЧерновик карточки:\n"
        + draft.model_dump_json(indent=2)
    )
    log.info("step_critique_start")
    return await call_structured(critic_agent(), prompt, CritiqueReport)


async def run_pipeline(supplier_text: str, max_attempts: int = 3):
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
    return draft, attempts, verdict
