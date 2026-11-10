from fastapi import APIRouter

from app.schemas import Card, JobIn
from app.services.pipeline import run_pipeline

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=Card)
async def create_card(body: JobIn) -> Card:
    draft, _attempts, _verdict = await run_pipeline(body.supplier_text)
    return draft
