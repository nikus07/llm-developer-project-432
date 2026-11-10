"""Роли: extractor → generator → critic (+ правщик одного поля). Промпт как код."""
import json

from agents import Agent, ModelSettings
from pydantic import BaseModel

from app.core.config import settings
from app.llm.output import RawJsonOutput
from app.schemas import Card, CritiqueReport, SupplierFacts

# Запас токенов под рассуждения: reasoning-модель тратит часть бюджета до ответа.
_MODEL_SETTINGS = ModelSettings(max_tokens=4096, temperature=0.4)


def _schema_text(model: type[BaseModel]) -> str:
    """Схема текстом: страховка для серверов без строгого режима."""
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


def extractor_agent() -> Agent:
    return Agent(
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
        model=settings.llm_model,
        model_settings=_MODEL_SETTINGS,
        output_type=RawJsonOutput(SupplierFacts),
    )


def generator_agent() -> Agent:
    return Agent(
        name="Generator",
        instructions=(
            "Ты — копирайтер карточек товара. По фактам о товаре составь черновик карточки.\n"
            "Верни СТРОГО JSON с этими полями и никаких других:\n"
            f"- title (string): заголовок, не длиннее {settings.card_title_max_len} символов\n"
            "- description (string): описание на 3-4 предложения\n"
            "- characteristics (object): пары ключ-значение, ТОЛЬКО из переданных фактов; "
            "пустых значений нет\n"
            "- benefits (array of string): 3-5 преимуществ\n"
            "- seo (object): meta_title, meta_description, keywords (array of string)\n"
            "- sources (array of object): для каждой характеристики {field, quote, document} — "
            "фрагмент исходного текста, откуда взято значение\n"
            "- missing_fields (array of string): чего в фактах нет; эти поля НЕ выдумывай и "
            "НЕ клади в characteristics\n"
            "- confidence (number 0..1): уверенность; чем больше недостающих полей, "
            "тем ниже\n"
            "Данных нет — положи имя поля в missing_fields, а не сочиняй значение. "
            "Если даны замечания критика — исправь именно их. "
            "Никакого markdown, только валидный JSON.\n"
            f"Схема ответа: {_schema_text(Card)}"
        ),
        model=settings.llm_model,
        model_settings=_MODEL_SETTINGS,
        output_type=RawJsonOutput(Card),
    )


def critic_agent() -> Agent:
    return Agent(
        name="Critic",
        instructions=(
            "Ты — критик карточек товара. Сравни черновик с фактами и проверь по правилам:\n"
            f"1. Заголовок не длиннее {settings.card_title_max_len} символов.\n"
            "2. В карточке нет характеристик, которых нет в фактах (выдумки).\n"
            "3. Описание и характеристики не пустые.\n"
            "4. Описание — 3-4 предложения, без воды.\n"
            "Верни СТРОГО JSON с этими полями и никаких других:\n"
            '- verdict (string): "approve" если все правила соблюдены, иначе "regenerate"\n'
            "- issues (array of string): нарушенные правила, по одному пункту на проблему "
            "(укажи номер правила); пустой список при approve\n"
            "Никакого markdown, только валидный JSON."
        ),
        model=settings.llm_model,
        model_settings=_MODEL_SETTINGS,
        output_type=RawJsonOutput(CritiqueReport),
    )


def field_fixer_agent() -> Agent:
    """Правит одно поле черновика. Без output_type: ответ — маленький JSON с одним ключом."""
    return Agent(
        name="FieldFixer",
        instructions=(
            "Ты исправляешь ровно одно поле черновика карточки товара. "
            "Верни только JSON-объект с этим полем и его новым значением. "
            "Никакого markdown и пояснений."
        ),
        model=settings.llm_model,
        model_settings=_MODEL_SETTINGS,
    )
