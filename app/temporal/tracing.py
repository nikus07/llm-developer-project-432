"""Воркер процессов привязывает идентификаторы задачи и запроса к контексту логов.

Идентификатор задачи уходит в процесс идентификатором самого процесса («card-<job_id>»),
идентификатор запроса лежит в payload задачи: перехватчик берёт оба и привязывает их
перед каждым шагом. Поэтому ни один шаг и ни один вызов модели внутри него не теряет их.
"""
import asyncio
import time

import structlog
from temporalio import activity
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
)

from app.core.logging import get_logger
from app.repositories import jobs as jobs_repo

log = get_logger(__name__)


def ids_from_workflow_id(workflow_id: str) -> dict[str, str]:
    """«card-<job_id>» → {job_id}; «doc-<document_id>» → {document_id}."""
    prefix, _, rest = workflow_id.partition("-")
    if not rest:
        return {}
    if prefix == "card":
        return {"job_id": rest}
    if prefix == "doc":
        return {"document_id": rest}
    return {}


class _TraceActivity(ActivityInboundInterceptor):
    async def execute_activity(self, input: ExecuteActivityInput):
        info = activity.info()
        context = {**ids_from_workflow_id(info.workflow_id), "workflow_id": info.workflow_id}
        job_id = context.get("job_id")
        if job_id:
            try:
                payload = await asyncio.to_thread(jobs_repo.get_payload, job_id)
                if payload and payload.get("request_id"):
                    context["request_id"] = payload["request_id"]
            except Exception as exc:  # трассировка не должна ронять шаг
                log.warning("trace_request_id_lookup_failed", error=repr(exc))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context)
        started = time.perf_counter()
        log.info("activity_started", activity=info.activity_type, attempt=info.attempt)
        try:
            return await super().execute_activity(input)
        except Exception as exc:
            log.error("activity_failed", activity=info.activity_type, error=repr(exc)[:300])
            raise
        finally:
            log.info(
                "activity_finished",
                activity=info.activity_type,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )


class TraceInterceptor(Interceptor):
    def intercept_activity(self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        return _TraceActivity(next)
