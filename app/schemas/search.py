from pydantic import BaseModel


class SearchHit(BaseModel):
    id: int
    document_id: str
    document: str
    idx: int
    kind: str
    page: int
    section: str
    article: str | None = None
    brand: str | None = None
    text: str
    score: float
