from fastapi import APIRouter, Header, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.repositories import documents as documents_repo
from app.repositories import jobs as jobs_repo
from app.schemas import GenerateIn, JobOut
from app.temporal.client import start_generate_workflow

router = APIRouter(tags=["generate"])
log = get_logger(__name__)


def _check_documents(document_ids: list[str]) -> None:
    documents = {i: documents_repo.get(i) for i in document_ids}
    unknown = [i for i, d in documents.items() if d is None]
    if unknown:
        raise HTTPException(status_code=404, detail=f"документы не найдены: {', '.join(unknown)}")
    failed = [
        f"{d['id']} ({d['filename']}): {d['reason']}"
        for d in documents.values()
        if d["status"] == "failed"
    ]
    if failed:
        raise HTTPException(status_code=422, detail="документы не разобраны: " + "; ".join(failed))


@router.post("/generate", response_model=JobOut, status_code=202)
async def generate(
    body: GenerateIn, idempotency_key: str | None = Header(default=None)
) -> dict:
    """Карточка по документам: 202 и идентификатор задачи, результат — GET /jobs/{id}."""
    document_ids = list(dict.fromkeys(body.document_ids))
    await run_in_threadpool(_check_documents, document_ids)

    payload = {"document_ids": document_ids, "hint": body.hint}
    job, created = await run_in_threadpool(jobs_repo.create, payload, idempotency_key)
    if created:
        try:
            await start_generate_workflow(job["id"], document_ids, body.hint)
        except Exception as exc:
            log.error("workflow_start_failed", job_id=job["id"], error=repr(exc))
            await run_in_threadpool(
                jobs_repo.update_status, job["id"], "failed", error="не удалось запустить процесс"
            )
            raise HTTPException(status_code=503, detail="движок процессов недоступен") from exc
    return job
