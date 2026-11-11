"""Резка блоков документа на фрагменты с метаданными.

Фрагмент — словарь: {"chunk_id", "ordinal", "text", "metadata"}. В метаданных документ,
страница, секция, тип (text | row) и, у строк таблицы с артикулом, артикул и бренд.
"""
from app.core.config import settings
from app.parsers.models import Block
from app.parsers.table import key_column, row_text


def _split_long_line(line: str, size: int) -> list[str]:
    """Строка длиннее размера фрагмента режется по словам."""
    pieces: list[str] = []
    current = ""
    for word in line.split(" "):
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Фиксированный размер с перекрытием; границы — только между строками.

    Перекрытие нужно, чтобы утверждение на границе не потерялось: хвост предыдущего
    фрагмента повторяется в начале следующего.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lines.extend(_split_long_line(line, size) if len(line) > size else [line])

    chunks: list[str] = []
    current: list[str] = []
    fresh = False  # есть ли в current что-то кроме повторённого хвоста
    length = 0
    for line in lines:
        if current and length + 1 + len(line) > size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) + 1 > overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous) + 1
            current, length, fresh = tail, tail_len, False
        current.append(line)
        length += len(line) + 1
        fresh = True
    if current and fresh:
        chunks.append("\n".join(current))
    return chunks


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Блоки → фрагменты. Текст режется с перекрытием, таблица — по строкам, строку не рвём.

    Строки таблицы с артикулом — по одной на фрагмент: артикул в метаданных точный, а
    поиск по артикулу возвращает строку, а не соседей. Остальные таблицы упаковываются
    строками до размера фрагмента.
    """
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    chunks: list[dict] = []

    def add(text: str, block: Block, kind: str, **extra) -> None:
        ordinal = len(chunks)
        metadata = {
            "doc_id": doc_id,
            "page": block.page,
            "section": block.section,
            "kind": kind,
            **{k: v for k, v in extra.items() if v},
        }
        chunks.append(
            {
                "chunk_id": f"{doc_id}-{ordinal}",
                "ordinal": ordinal,
                "text": text,
                "metadata": metadata,
            }
        )

    for block in blocks:
        if block.kind == "text":
            for piece in split_text(block.content, size, overlap):
                add(piece, block, "text")
            continue

        headers, body = block.rows[0], block.rows[1:]
        rows = [(row_text(headers, row), row) for row in body]
        rows = [(parts, row) for parts, row in rows if parts[0]]
        if key_column(headers) is not None:
            for (text, articul, brand), _row in rows:
                add(text, block, "row", articul=articul, brand=brand)
            continue

        batch: list[str] = []
        length = 0
        for (text, _articul, _brand), _row in rows:
            if batch and length + 1 + len(text) > size:
                add("\n".join(batch), block, "row")
                batch, length = [], 0
            batch.append(text)
            length += len(text) + 1
        if batch:
            add("\n".join(batch), block, "row")
    return chunks
