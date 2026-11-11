from pydantic import BaseModel


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    document: str
    page: int | None = None
    section: str | None = None
    kind: str | None = None
    articul: str | None = None
    brand: str | None = None
    text: str
    score: float
