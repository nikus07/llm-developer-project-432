"""docx: параграфы и таблицы; заголовок — граница секции."""
from io import BytesIO

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.parsers.models import Block, DocumentRejected
from app.parsers.normalizer import clean_text
from app.parsers.table import table_block

_DEFAULT_SECTION = "Документ"
_HEADING_PREFIXES = ("heading", "заголовок", "title")
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _is_heading(paragraph: Paragraph) -> bool:
    name = (paragraph.style.name if paragraph.style is not None else "").lower()
    return name.startswith(_HEADING_PREFIXES)


def _page_breaks(paragraph: Paragraph) -> int:
    """Явные разрывы страниц и метки, которые Word оставляет при сохранении."""
    element = paragraph._p
    explicit = sum(1 for br in element.iter(f"{_W}br") if br.get(f"{_W}type") == "page")
    rendered = sum(1 for _ in element.iter(f"{_W}lastRenderedPageBreak"))
    return max(explicit, rendered)


def parse_docx(data: bytes) -> list[Block]:
    try:
        document = Document(BytesIO(data))
    except Exception as exc:
        raise DocumentRejected(f"не удалось открыть docx: {exc}") from exc

    blocks: list[Block] = []
    section = _DEFAULT_SECTION
    page = 1
    buffer: list[str] = []
    buffer_page = page

    def flush() -> None:
        nonlocal buffer
        text = clean_text("\n".join(buffer))
        if text:
            blocks.append(Block(kind="text", content=text, page=buffer_page, section=section))
        buffer = []

    for child in document.element.body.iterchildren():
        if child.tag == f"{_W}p":
            paragraph = Paragraph(child, document)
            page += _page_breaks(paragraph)
            text = paragraph.text.strip()
            if _is_heading(paragraph) and text:
                flush()
                section = text
                buffer_page = page
            elif text:
                if not buffer:
                    buffer_page = page
                buffer.append(text)
        elif child.tag == f"{_W}tbl":
            flush()
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            block = table_block(rows, page=page, section=section)
            if block is not None:
                blocks.append(block)
            buffer_page = page
    flush()
    return blocks
