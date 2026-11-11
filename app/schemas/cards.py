"""Контракты результатов: карточка и промежуточные ответы агентов."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings


class SupplierFacts(BaseModel):
    """Результат extractor-агента: структурированные факты из текста поставщика."""

    product_name: str = Field(description="Название товара из текста поставщика")
    characteristics: dict[str, str] = Field(
        default_factory=dict, description="Характеристики: ключ-значение"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Что не найдено в тексте (например: гарантия, цвет)",
    )


class SourceRef(BaseModel):
    """Откуда взято значение: фрагмент документа, на который ссылается карточка."""

    chunk_id: str = Field(description="id фрагмента из заголовка источника в контексте")
    doc_id: str = ""
    page: int | None = None
    score: float | None = None
    # необязательные подробности: какая характеристика опирается на источник и откуда выдержка
    field: str = Field(default="", description="Характеристика, для которой источник")
    quote: str = Field(default="", description="Дословная выдержка из источника")
    section: str | None = Field(default=None, description="Секция; заполняет приложение")
    document: str | None = Field(default=None, description="Имя файла; заполняет приложение")

    @field_validator("chunk_id", mode="before")
    @classmethod
    def _chunk_id_as_text(cls, value):
        return str(value) if isinstance(value, int) else value


class Seo(BaseModel):
    title: str = ""
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class CardDraft(BaseModel):
    """Результат generator-агента: черновик карточки товара."""

    title: str = Field(min_length=1, description="Заголовок карточки")
    description: str = Field(description="Описание на 3-4 предложения")
    characteristics: dict[str, str] = Field(
        default_factory=dict, description="Характеристики только из фактов, без пустых значений"
    )
    benefits: list[str] = Field(default_factory=list)
    seo: Seo = Field(default_factory=Seo)
    sources: list[SourceRef] = Field(
        default_factory=list, description="Источник для каждой характеристики"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Чего в документах не нашлось; такие поля нельзя выдумывать",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Уверенность в карточке от 0 до 1"
    )

    @field_validator("title")
    @classmethod
    def _title_is_short(cls, value: str) -> str:
        limit = settings.card_title_max_len
        if len(value) > limit:
            raise ValueError(f"заголовок {len(value)} символов, максимум {limit}: сократи")
        return value

    @field_validator("characteristics")
    @classmethod
    def _drop_empty_values(cls, value: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in value.items() if v is not None and str(v).strip()}

    @model_validator(mode="after")
    def _field_is_not_in_both_lists(self) -> "CardDraft":
        both = sorted(set(self.characteristics) & set(self.missing_fields))
        if both:
            raise ValueError(
                f"поля {both} одновременно в characteristics и в missing_fields"
            )
        return self


class CritiqueReport(BaseModel):
    """Результат critic-агента: вердикт и список проблем."""

    verdict: Literal["approve", "regenerate"] = Field(
        description="approve — карточка годится; regenerate — переделать"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Список проблем: заголовок длиннее допустимого, "
        "выдуманные характеристики, пустые поля и т.п.",
    )


class JudgeVerdict(BaseModel):
    """Вердикт модели-судьи: подтверждает ли набор источников утверждения карточки."""

    unsupported_characteristics: list[str] = Field(
        default_factory=list,
        description="Ключи характеристик, значения которых источники не подтверждают",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Утверждения из описания и преимуществ без опоры на источники",
    )
    comment: str = ""
