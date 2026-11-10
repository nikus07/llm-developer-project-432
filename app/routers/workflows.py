"""Управление процессом генерации: состояние и сигналы подтверждения/отказа."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from temporalio.service import RPCError, RPCStatusCode

from app.temporal.client import get_client, workflow_id

router = APIRouter(prefix="/workflows", tags=["workflows"])


class RejectIn(BaseModel):
    reason: str = ""


async def _handle(job_id: str):
    client = await get_client()
    return client.get_workflow_handle(workflow_id(job_id))


def _not_found(exc: RPCError) -> HTTPException:
    if exc.status == RPCStatusCode.NOT_FOUND:
        return HTTPException(status_code=404, detail="процесс не найден")
    return HTTPException(status_code=502, detail=f"движок процессов: {exc.message}")


@router.get("/{job_id}")
async def workflow_state(job_id: str) -> dict:
    handle = await _handle(job_id)
    try:
        return await handle.query("state")
    except RPCError as exc:
        raise _not_found(exc) from exc


@router.post("/{job_id}/approve")
async def approve(job_id: str) -> dict:
    handle = await _handle(job_id)
    try:
        await handle.signal("approve")
    except RPCError as exc:
        raise _not_found(exc) from exc
    return {"status": "signal_sent"}


@router.post("/{job_id}/reject")
async def reject(job_id: str, body: RejectIn | None = None) -> dict:
    handle = await _handle(job_id)
    try:
        await handle.signal("reject", body.reason if body else "")
    except RPCError as exc:
        raise _not_found(exc) from exc
    return {"status": "signal_sent"}
