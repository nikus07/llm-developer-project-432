"""Стадии пайплайна. Один код для FastAPI, Celery и Temporal."""
import re

from app.agents.prompts import critic_agent, extractor_agent, generator_agent
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas import CardDraft, CritiqueReport, SupplierFacts

log = get_logger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _ensure_json(text: str) -> str:
    """Срезает обёртку ```json ... ```, которую добавляют локальные модели."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


async def extract(supplier_text: str) -> SupplierFacts:
    log.info("step_extract_start")
    out = await run_agent(extractor_agent(), supplier_text)
    if isinstance(out, SupplierFacts):
        return out
    return SupplierFacts.model_validate_json(_ensure_json(out))


async def generate(
    facts: SupplierFacts, feedback: list[str] | None = None
) -> CardDraft:
    prompt = "Факты о товаре:\n" + facts.model_dump_json(indent=2)
    if feedback:
        prompt += "\n\nЗамечания критика, учти:\n" + "\n".join(
            f"- {i}" for i in feedback
        )
    log.info("step_generate_start", feedback=bool(feedback))
    out = await run_agent(generator_agent(), prompt)
    if isinstance(out, CardDraft):
        return out
    return CardDraft.model_validate_json(_ensure_json(out))


async def critique(facts: SupplierFacts, draft: CardDraft) -> CritiqueReport:
    prompt = (
        "Факты о товаре:\n"
        + facts.model_dump_json(indent=2)
        + "\n\nЧерновик карточки:\n"
        + draft.model_dump_json(indent=2)
    )
    log.info("step_critique_start")
    out = await run_agent(critic_agent(), prompt)
    if isinstance(out, CritiqueReport):
        return out
    return CritiqueReport.model_validate_json(_ensure_json(out))


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
