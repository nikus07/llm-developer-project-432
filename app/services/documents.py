"""Сервис документа: разбор → нормализация → резка → сохранение."""
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.parsers import docx, pdf, xlsx
from app.parsers.chunker import chunk_blocks
from app.parsers.models import Chunk, DocumentRejected
from app.rag.reindex import index_chunks
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo

log = get_logger(__name__)

PARSERS = {".pdf": pdf.parse_pdf, ".docx": docx.parse_docx, ".xlsx": xlsx.parse_xlsx}
SUPPORTED_EXTENSIONS = frozenset(PARSERS)


def stored_path(document_id: str, filename: str) -> Path:
    return settings.upload_dir / f"{document_id}{Path(filename).suffix.lower()}"


def save_upload(document_id: str, filename: str, content: bytes) -> Path:
    path = stored_path(document_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def parse_to_chunks(path: Path | str) -> list[Chunk]:
    """Файл → разбор → нормализация → резка: фрагменты с метаданными, без базы.

    Расширение выбирает разборщик. Документ, который разобрать нельзя (скан без текстового
    слоя, битый файл, пустой результат), даёт DocumentRejected с причиной, а не пустой список.
    """
    path = Path(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise DocumentRejected(f"формат {path.suffix or 'без расширения'} не поддерживается")
    chunks = chunk_blocks(parser(path))
    if not chunks:
        raise DocumentRejected("из документа не удалось извлечь ни одного фрагмента")
    return chunks


def process_document(document_id: str) -> dict:
    """Блокирующий шаг: диск, разбор файла, база. Исполняется в потоке воркера.

    Состояния: new → parsing → parsed (фрагменты в базе) → indexed (есть векторы, ищется).
    Ожидаемый отказ (скан, битый файл) — статус «failed» с причиной, а не тихий пустой
    результат. Неожиданная ошибка уходит наверх: движок повторит шаг.
    """
    document = documents_repo.get(document_id)
    if document is None:
        raise LookupError(f"документ {document_id} не найден")

    documents_repo.set_status(document_id, "parsing")
    try:
        chunks = parse_to_chunks(stored_path(document_id, document["filename"]))
    except DocumentRejected as exc:
        log.warning("document_rejected", document_id=document_id, reason=str(exc))
        documents_repo.set_status(document_id, "failed", reason=str(exc))
        return {"status": "failed", "reason": str(exc), "chunks": 0}

    chunks_repo.replace_for_document(document_id, document["filename"], chunks)
    pages = max(chunk.page for chunk in chunks)
    documents_repo.set_status(document_id, "parsed", pages=pages)
    log.info("document_parsed", document_id=document_id, chunks=len(chunks))
    return {"status": "parsed", "reason": None, "chunks": len(chunks)}


def index_document(document_id: str) -> dict:
    """Продолжение цепочки: векторы фрагментам разобранного документа → «indexed».

    Блокирующий шаг (модель на процессоре, база): исполняется в потоке воркера.
    Повторный запуск не пересчитывает уже посчитанные векторы.
    """
    indexed = index_chunks(document_id)
    documents_repo.set_status(document_id, "indexed")
    log.info("document_indexed", document_id=document_id, new_vectors=indexed)
    return {"status": "indexed", "new_vectors": indexed}
