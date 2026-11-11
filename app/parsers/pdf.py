"""pdf: текст постранично; крупный жирный шрифт — заголовок секции."""
import re
from pathlib import Path

from pypdf import PdfReader

from app.parsers.models import Block, DocumentRejected
from app.parsers.normalizer import (
    clean_text,
    edge_lines,
    is_page_marker,
    line_key,
    repeated_line_keys,
)


class PdfNoTextLayer(DocumentRejected):
    """В pdf нет текстового слоя: скан. Распознавание текста не подключено."""


_DEFAULT_SECTION = "Документ"
_HEADING_MIN_SIZE = 11.0
# в режиме layout колонки таблицы разделены широким просветом
_COLUMN_GAP_RE = re.compile(r"\s{3,}")


def _heading_texts(page) -> set[str]:
    """Тексты крупного жирного шрифта: заголовки секций."""
    found: set[str] = set()

    def visitor(text, _cm, _tm, font_dict, font_size):
        font = str((font_dict or {}).get("/BaseFont", "")).lower()
        if text.strip() and font_size >= _HEADING_MIN_SIZE and "bold" in font:
            found.add(text.strip())

    page.extract_text(visitor_text=visitor)
    return found


def _page_lines(page) -> list[str]:
    layout = page.extract_text(extraction_mode="layout") or ""
    lines: list[str] = []
    for raw in layout.split("\n"):
        line = raw.strip()
        if not line:
            continue
        columns = _COLUMN_GAP_RE.split(line)
        # две колонки — пара «параметр: значение», чтобы строка читалась целиком
        lines.append(f"{columns[0]}: {columns[1]}" if len(columns) == 2 else " ".join(columns))
    return lines


def parse_pdf(path: Path) -> list[Block]:
    try:
        reader = PdfReader(str(path))
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

    repeated = repeated_line_keys([lines for lines, _ in pages])
    for lines, _ in pages:
        repeated |= {line_key(ln) for ln in edge_lines(lines) if is_page_marker(ln)}
    blocks: list[Block] = []
    section = _DEFAULT_SECTION
    for number, (lines, headings) in enumerate(pages, start=1):
        buffer: list[str] = []

        def flush(page_number: int, buf: list[str], title: str) -> None:
            text = clean_text("\n".join(buf))
            if text:
                blocks.append(Block(kind="text", text=text, page=page_number, section=title))

        for line in lines:
            if line_key(line) in repeated:
                continue  # колонтитул
            if line in headings:
                flush(number, buffer, section)
                buffer = []
                section = line
            else:
                buffer.append(line)
        flush(number, buffer, section)
    return blocks
