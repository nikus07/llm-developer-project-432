"""Три агента: extractor → generator → critic. Промпт как код."""
from agents import Agent, AgentOutputSchema, ModelSettings

from app.core.config import settings
from app.schemas import CardDraft, CritiqueReport, SupplierFacts

_MODEL = settings.llm_model
# Запас токенов под рассуждения: reasoning-модель тратит часть бюджета до ответа.
_MODEL_SETTINGS = ModelSettings(max_tokens=4096, temperature=0.4)

extractor_agent = Agent(
    name="Extractor",
    instructions=(
        "Ты — извлекатель фактов. Из текста поставщика выдели структурированные факты.\n"
        "Верни СТРОГО JSON с этими полями и никаких других:\n"
        "- product_name (string): название товара\n"
        "- characteristics (object): пары ключ-значение, ТОЛЬКО из текста\n"
        "- missing_fields (array of string): что важно для карточки, но в тексте нет "
        "(например: гарантия, цвет, страна-производитель)\n"
        "Не выдумывай. Нет данных — поле в missing_fields. "
        "Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(SupplierFacts, strict_json_schema=False),
)

generator_agent = Agent(
    name="Generator",
    instructions=(
        "Ты — копирайтер карточек товара. По фактам о товаре составь черновик карточки.\n"
        "Верни СТРОГО JSON с этими полями и никаких других:\n"
        "- title (string): заголовок, не длиннее 60 символов\n"
        "- description (string): описание на 3-4 предложения\n"
        "- characteristics (object): пары ключ-значение, ТОЛЬКО из переданных фактов\n"
        "- benefits (array of string): 3-5 преимуществ\n"
        "Не выдумывай характеристик, которых нет в фактах. Если даны замечания критика — "
        "исправь именно их. Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(CardDraft, strict_json_schema=False),
)

critic_agent = Agent(
    name="Critic",
    instructions=(
        "Ты — критик карточек товара. Сравни черновик с фактами и проверь по правилам:\n"
        "1. Заголовок не длиннее 60 символов.\n"
        "2. В карточке нет характеристик, которых нет в фактах (выдумки).\n"
        "3. Описание и характеристики не пустые.\n"
        "4. Описание — 3-4 предложения, без воды.\n"
        "Верни СТРОГО JSON с этими полями и никаких других:\n"
        '- verdict (string): "approve" если все правила соблюдены, иначе "regenerate"\n'
        "- issues (array of string): нарушенные правила, по одному пункту на проблему "
        "(укажи номер правила); пустой список при approve\n"
        "Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(CritiqueReport, strict_json_schema=False),
)
