from pydantic import BaseModel

from app.schemas.cards import Card


class JobIn(BaseModel):
    supplier_text: str


class JobOut(BaseModel):
    id: str
    status: str
    attempts: int
    result: Card | None = None
    error: str | None = None
