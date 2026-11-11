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
    """Откуда взято значение: поле карточки и фрагмент документа."""

    field: str = Field(description="Поле или характеристика карточки")
    quote: str = Field(description="Фрагмент документа, на который опирается значение")
    document: str | None = Field(default=None, description="Имя документа, если известно")
    chunk_id: int | None = Field(
        default=None, description="id фрагмента из атрибута id тега source"
    )
    page: int | None = Field(default=None, description="Страница; заполняет приложение")
    section: str | None = Field(default=None, description="Секция; заполняет приложение")


class Seo(BaseModel):
    meta_title: str = ""
    meta_description: str = ""
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
        default=0.0, ge=0.0, le=1.0, description="Уверенность в карточке от 0 до 1"
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
