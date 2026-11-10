"""Резка блоков документа на фрагменты с метаданными."""
from app.core.config import settings
from app.parsers.models import Block, Chunk


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


def _join(values: list[str | None]) -> str | None:
    unique = list(dict.fromkeys(v for v in values if v))
    return ", ".join(unique) if unique else None


def chunk_blocks(
    blocks: list[Block],
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    chunks: list[Chunk] = []

    def add(kind: str, text: str, block: Block, article=None, brand=None) -> None:
        chunks.append(
            Chunk(
                idx=len(chunks),
                kind=kind,
                text=text,
                page=block.page,
                section=block.section,
                article=article,
                brand=brand,
            )
        )

    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.kind == "text":
            for piece in split_text(block.text, size, overlap):
                add("text", piece, block)
            index += 1
            continue

        # таблица: подряд идущие строки одной секции и страницы; строку не рвём
        group = [block]
        index += 1
        while (
            index < len(blocks)
            and blocks[index].kind == "row"
            and (blocks[index].page, blocks[index].section) == (block.page, block.section)
        ):
            group.append(blocks[index])
            index += 1

        batch: list[Block] = []
        length = 0
        for row in group:
            if batch and length + 1 + len(row.text) > size:
                add(
                    "row",
                    "\n".join(b.text for b in batch),
                    block,
                    _join([b.article for b in batch]),
                    _join([b.brand for b in batch]),
                )
                batch, length = [], 0
            batch.append(row)
            length += len(row.text) + 1
        if batch:
            add(
                "row",
                "\n".join(b.text for b in batch),
                block,
                _join([b.article for b in batch]),
                _join([b.brand for b in batch]),
            )
    return chunks
