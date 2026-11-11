from fastapi import APIRouter

from app.schemas import CardDraft, JobIn
from app.services.pipeline import run_pipeline

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=CardDraft)
async def create_card(body: JobIn) -> CardDraft:
    draft, _attempts, _verdict, _status = await run_pipeline(body.supplier_text)
    return draft
