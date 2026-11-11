"""Модель в роли судьи: подтверждают ли источники утверждения карточки."""
from app.agents.prompts import judge_agent
from app.schemas import CardDraft, JudgeVerdict
from app.services.structured import call_structured

# судье нужны только утверждения; ссылки, уверенность и список недостающих его бы подсказали
_HIDDEN = {"sources", "confidence", "missing_fields"}


def judge_prompt(context: str, card: CardDraft) -> str:
    """Фрагменты и карточка: судья сам сверяет значения с текстом источников."""
    return (
        "Источники:\n"
        + context
        + "\n\nКарточка (JSON):\n"
        + card.model_dump_json(indent=2, exclude=_HIDDEN)
    )


async def judge_card(context: str, card: CardDraft) -> JudgeVerdict:
    return await call_structured(judge_agent(), judge_prompt(context, card), JudgeVerdict)
