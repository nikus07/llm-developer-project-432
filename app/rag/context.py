"""Контекст генерации: найденные фрагменты одной строкой с опознаваемыми заголовками."""
import re

from app.core.config import settings

_SPACES_RE = re.compile(r"\s+")
# заголовок источника: «[id | файл | стр. N | секция]»; id — то, на что ссылается модель
_HEADER_RE = re.compile(r"^\[([^\s|\]]+)(?: \|[^\]\n]*)?\]$", re.MULTILINE)


def _fingerprint(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip().casefold()


def _safe(text: str) -> str:
    """Строка текста, похожая на заголовок источника, сдвигается: id нельзя подделать."""
    return "\n".join(" " + line if line.startswith("[") else line for line in text.split("\n"))


def _header(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    parts = [str(chunk["chunk_id"]), str(chunk.get("filename") or metadata.get("doc_id") or "")]
    if metadata.get("page") is not None:
        parts.append(f"стр. {metadata['page']}")
    if metadata.get("section"):
        parts.append(str(metadata["section"]).replace("]", ")").replace("\n", " "))
    return "[" + " | ".join(parts) + "]"


def build_context(chunks: list[dict], max_chars: int | None = None) -> str:
    """Контекст из найденных фрагментов в порядке релевантности.

    - одинаковый текст (найденный дважды или повторённый в разных документах) берётся один раз;
    - размер ограничен: фрагмент, который не помещается, пропускается — меньшие, что идут
      дальше, ещё могут поместиться.
    """
    limit = max_chars or settings.context_max_chars
    blocks: list[str] = []
    seen: set[str] = set()
    used = 0
    for chunk in chunks:
        fingerprint = _fingerprint(chunk["text"])
        if fingerprint in seen:
            continue
        block = f"{_header(chunk)}\n{_safe(chunk['text'])}"
        size = len(block) + 2  # разделитель между источниками
        if used + size > limit:
            continue
        seen.add(fingerprint)
        blocks.append(block)
        used += size
    return "\n\n".join(blocks)


def context_chunk_ids(context: str) -> set[str]:
    """Идентификаторы фрагментов, которые были в контексте, поданном модели."""
    return {m.group(1) for m in _HEADER_RE.finditer(context)}
