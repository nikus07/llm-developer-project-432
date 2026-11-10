from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str  # new | parsing | indexed | failed
    reason: str | None = None
    pages: int | None = None
    chunks: int = 0
