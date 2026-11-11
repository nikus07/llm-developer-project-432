"""pdf: текст и таблицы постранично; крупный жирный шрифт — заголовок секции."""
import re
from io import BytesIO

from pypdf import PdfReader

from app.parsers.models import Block, DocumentRejected
from app.parsers.table import table_block

_DEFAULT_SECTION = "Документ"
_HEADING_MIN_SIZE = 11.0
# в режиме layout колонки таблицы разделены широким просветом
_COLUMN_GAP_RE = re.compile(r"\s{3,}")


class PdfNoTextLayer(DocumentRejected):
    """В pdf нет текстового слоя: скан. Распознавание текста не подключено."""


def _heading_texts(page) -> set[str]:
    """Тексты крупного жирного шрифта: заголовки секций."""
    found: set[str] = set()

    def visitor(text, _cm, _tm, font_dict, font_size):
        font = str((font_dict or {}).get("/BaseFont", "")).lower()
        if text.strip() and font_size >= _HEADING_MIN_SIZE and "bold" in font:
            found.add(text.strip())

    page.extract_text(visitor_text=visitor)
    return found


def _page_lines(page) -> list[list[str]]:
    """Строки страницы, каждая — список колонок (одна колонка — обычный текст)."""
    layout = page.extract_text(extraction_mode="layout") or ""
    rows: list[list[str]] = []
    for raw in layout.split("\n"):
        line = raw.strip()
        if line:
            rows.append(_COLUMN_GAP_RE.split(line))
    return rows


def parse_pdf(data: bytes) -> list[Block]:
    """pdf → блоки. Таблица — подряд идущие строки с одним числом колонок (две и больше).

    Колонтитулы здесь не удаляются: их видно только на фоне других страниц, поэтому их
    убирает нормализатор (parsers.normalizer.normalize).
    """
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise DocumentRejected("pdf защищён паролем")
        pages = [(_page_lines(p), _heading_texts(p)) for p in reader.pages]
    except DocumentRejected:
        raise
    except Exception as exc:
        raise DocumentRejected(f"не удалось открыть pdf: {exc}") from exc

    if not any(lines for lines, _ in pages):
        raise PdfNoTextLayer(
            "в pdf нет текстового слоя (похоже на скан): распознавание текста не подключено"
        )

    blocks: list[Block] = []
    section = _DEFAULT_SECTION
    text: list[str] = []
    table: list[list[str]] = []

    def flush_text(page: int, title: str) -> None:
        if text:
            blocks.append(
                Block(kind="text", content="\n".join(text), page=page, section=title)
            )
            text.clear()

    def flush_table(page: int, title: str) -> None:
        if len(table) >= 2:
            block = table_block(table, page=page, section=title)
            if block is not None:
                blocks.append(block)
        else:  # одна «табличная» строка — обычный текст
            text.extend(" ".join(row) for row in table)
        table.clear()

    for number, (rows, headings) in enumerate(pages, start=1):
        for row in rows:
            if len(row) >= 2:
                if table and len(row) != len(table[0]):
                    flush_table(number, section)  # другое число колонок — новая таблица
                if not table:
                    flush_text(number, section)
                table.append(row)
                continue
            if table:
                flush_table(number, section)
            line = " ".join(row)
            if line in headings:
                flush_text(number, section)
                section = line
            else:
                text.append(line)
        if table:
            flush_table(number, section)
        flush_text(number, section)
    return blocks
