from fastapi import APIRouter, Header, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.repositories import jobs as jobs_repo
from app.schemas import JobIn, JobOut
from app.temporal.client import start_card_workflow

router = APIRouter(tags=["jobs"])
log = get_logger(__name__)


@router.post("/jobs", response_model=JobOut, status_code=202)
async def create_job(
    body: JobIn, idempotency_key: str | None = Header(default=None)
) -> dict:
    job, created = await run_in_threadpool(
        jobs_repo.create, {"supplier_text": body.supplier_text}, idempotency_key
    )
    if created:
        try:
            await start_card_workflow(job["id"], body.supplier_text)
        except Exception as exc:
            log.error("workflow_start_failed", job_id=job["id"], error=repr(exc))
            await run_in_threadpool(
                jobs_repo.update_status, job["id"], "failed", error="не удалось запустить процесс"
            )
            raise HTTPException(status_code=503, detail="движок процессов недоступен") from exc
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str) -> dict:
    job = await run_in_threadpool(jobs_repo.get, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    return job
