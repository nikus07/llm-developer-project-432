from decimal import Decimal

from pydantic import BaseModel, Field

from app.schemas.cards import Card


class JobIn(BaseModel):
    supplier_text: str


class GenerateIn(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=20, description="Документы")
    hint: str | None = Field(default=None, max_length=500, description="Что это за товар")


class JobOut(BaseModel):
    id: str
    status: str
    attempts: int
    result: Card | None = None
    error: str | None = None
    report: dict | None = None  # отчёт защиты: замаскировано и исключено


class JobCost(BaseModel):
    job_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost: Decimal  # ₽, условные цены из настроек; строкой, не float
    duration_ms: int
