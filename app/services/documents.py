"""Сервис документа: разбор → нормализация → резка → сохранение."""
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.parsers.chunker import chunk_blocks
from app.parsers.docx import parse_docx
from app.parsers.models import DocumentRejected
from app.parsers.normalizer import normalize
from app.parsers.pdf import parse_pdf
from app.parsers.xlsx import parse_xlsx
from app.rag.reindex import index_chunks
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo

log = get_logger(__name__)

PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx}
SUPPORTED_EXTENSIONS = frozenset(f".{kind}" for kind in PARSERS)


def kind_of(filename: str) -> str:
    """Тип документа по расширению имени файла: pdf | docx | xlsx."""
    return Path(filename).suffix.lower().lstrip(".")


def stored_path(document_id: str, filename: str) -> Path:
    return settings.upload_dir / f"{document_id}{Path(filename).suffix.lower()}"


def save_upload(document_id: str, filename: str, content: bytes) -> Path:
    path = stored_path(document_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def parse_to_chunks(data: bytes, kind: str, doc_id: str) -> list[dict]:
    """Файл (байты) → разбор → нормализация → резка: фрагменты с метаданными, без базы.

    kind — pdf | docx | xlsx. Документ, который разобрать нельзя (скан без текстового слоя,
    битый файл, пустой результат), даёт DocumentRejected с причиной, а не пустой список.
    """
    parser = PARSERS.get(kind.lower().lstrip("."))
    if parser is None:
        raise DocumentRejected(f"формат {kind or 'без расширения'} не поддерживается")
    chunks = chunk_blocks(normalize(parser(data)), doc_id)
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
        content = stored_path(document_id, document["filename"]).read_bytes()
        chunks = parse_to_chunks(content, document["kind"], document_id)
    except DocumentRejected as exc:
        log.warning("document_rejected", document_id=document_id, reason=str(exc))
        documents_repo.set_status(document_id, "failed", reason=str(exc))
        return {"status": "failed", "reason": str(exc), "chunks": 0}

    chunks_repo.replace_for_document(document_id, chunks)
    pages = max(chunk["metadata"]["page"] for chunk in chunks)
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
