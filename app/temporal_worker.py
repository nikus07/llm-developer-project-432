"""Точка входа Temporal-воркера: регистрирует workflows + activities."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from app.core.config import settings
from app.core.db import close_pool, init_pool
from app.core.logging import get_logger, setup_logging
from app.temporal.activities import (
    critique_activity,
    documents_state_activity,
    extract_activity,
    generate_activity,
    index_document_activity,
    mark_document_failed_activity,
    output_filter_activity,
    process_document_activity,
    rag_critique_activity,
    rag_extract_activity,
    rag_generate_activity,
    retrieve_activity,
    screen_activity,
    set_status_activity,
    verify_references_activity,
)
from app.temporal.tracing import TraceInterceptor
from app.temporal.workflows import CardWorkflow, DocumentWorkflow, GenerateWorkflow

log = get_logger(__name__)


async def main():
    setup_logging()
    init_pool()  # пул нужен шагу записи статуса
    client = await Client.connect(settings.temporal_host)
    # блокирующие шаги (база) движок берёт только при выданном пуле потоков
    with ThreadPoolExecutor(max_workers=8) as executor:
        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=[CardWorkflow, DocumentWorkflow, GenerateWorkflow],
            activities=[
                extract_activity,
                generate_activity,
                critique_activity,
                set_status_activity,
                process_document_activity,
                index_document_activity,
                mark_document_failed_activity,
                documents_state_activity,
                rag_critique_activity,
                rag_extract_activity,
                rag_generate_activity,
                retrieve_activity,
                verify_references_activity,
                screen_activity,
                output_filter_activity,
            ],
            activity_executor=executor,
            interceptors=[TraceInterceptor()],
        )
        log.info("temporal_worker_started", task_queue=settings.temporal_task_queue)
        try:
            await worker.run()
        finally:
            close_pool()


if __name__ == "__main__":
    asyncio.run(main())
