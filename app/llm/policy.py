"""Политика двух моделей: служебные вызовы идут на дешёвую, генерация — на основную."""
from decimal import Decimal
from typing import Literal

from app.core.config import settings

Tier = Literal["main", "light"]

# Служебные вызовы не требуют сильной модели: проверка черновика, правка одного поля,
# судья. Основная генерация (извлечение фактов и сборка карточки) требует.
LIGHT_AGENTS = frozenset({"Critic", "RagCritic", "FieldFixer", "Judge"})


def tier_for(agent_name: str) -> Tier:
    return "light" if agent_name in LIGHT_AGENTS else "main"


def model_for(tier: Tier) -> str:
    if tier == "light":
        return settings.llm_model_light or settings.llm_model
    return settings.llm_model


def price_for(tier: Tier) -> tuple[Decimal, Decimal]:
    """(цена входных, цена выходных токенов) за миллион, в ₽."""
    if tier == "light":
        return settings.llm_price_light_input, settings.llm_price_light_output
    return settings.llm_price_main_input, settings.llm_price_main_output
