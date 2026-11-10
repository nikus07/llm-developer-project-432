import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.logging import get_logger
from app.repositories import documents as documents_repo
from app.schemas.documents import DocumentOut
from app.services.documents import SUPPORTED_EXTENSIONS, save_upload
from app.temporal.client import start_document_workflow

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)


@router.post("", response_model=DocumentOut, status_code=202)
async def upload_document(file: UploadFile) -> dict:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"формат {extension or 'без расширения'} не поддерживается: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"файл больше {settings.max_upload_mb} МБ")
    if not content:
        raise HTTPException(status_code=400, detail="файл пустой")

    content_hash = hashlib.sha256(content).hexdigest()
    document, created = await run_in_threadpool(
        documents_repo.create, filename, content_hash, len(content)
    )
    if created:
        try:
            await run_in_threadpool(save_upload, document["id"], filename, content)
            await start_document_workflow(document["id"])
        except Exception as exc:
            log.error("document_start_failed", document_id=document["id"], error=repr(exc))
            await run_in_threadpool(
                documents_repo.set_status,
                document["id"],
                "failed",
                reason="не удалось запустить разбор",
            )
            raise HTTPException(status_code=503, detail="разбор документа не запущен") from exc
    return document


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str) -> dict:
    document = await run_in_threadpool(documents_repo.get, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="документ не найден")
    return document
