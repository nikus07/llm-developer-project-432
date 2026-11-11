from temporalio.client import Client

from app.core.config import settings

_client: Client | None = None


def workflow_id(job_id: str) -> str:
    return f"card-{job_id}"


async def get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(settings.temporal_host)
    return _client


async def start_card_workflow(job_id: str, supplier_text: str) -> None:
    from app.temporal.workflows import CardWorkflow

    client = await get_client()
    await client.start_workflow(
        CardWorkflow.run,
        args=[job_id, supplier_text, settings.confidence_threshold],
        id=workflow_id(job_id),
        task_queue=settings.temporal_task_queue,
    )


def document_workflow_id(document_id: str) -> str:
    return f"doc-{document_id}"


async def start_document_workflow(document_id: str) -> None:
    from app.temporal.workflows import DocumentWorkflow

    client = await get_client()
    await client.start_workflow(
        DocumentWorkflow.run,
        document_id,
        id=document_workflow_id(document_id),
        task_queue=settings.temporal_task_queue,
    )


async def start_generate_workflow(job_id: str, document_ids: list[str], hint: str | None) -> None:
    from app.temporal.workflows import GenerateWorkflow

    client = await get_client()
    await client.start_workflow(
        GenerateWorkflow.run,
        args=[job_id, document_ids, hint, settings.confidence_threshold],
        id=workflow_id(job_id),
        task_queue=settings.temporal_task_queue,
    )
