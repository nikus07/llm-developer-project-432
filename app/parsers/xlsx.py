"""xlsx: спецификация «артикул → параметры», заголовки колонок — первая строка."""
from pathlib import Path

from openpyxl import load_workbook

from app.parsers.models import Block, DocumentRejected
from app.parsers.table import cell_text, row_to_block


def parse(path: Path) -> list[Block]:
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise DocumentRejected(f"не удалось открыть xlsx: {exc}") from exc

    blocks: list[Block] = []
    for sheet_number, sheet in enumerate(workbook.worksheets, start=1):
        rows = [list(row) for row in sheet.iter_rows(values_only=True)]
        rows = [row for row in rows if any(cell_text(c) for c in row)]
        if len(rows) < 2:
            continue
        headers = [cell_text(c) for c in rows[0]]
        for values in rows[1:]:
            block = row_to_block(headers, values, page=sheet_number, section=sheet.title)
            if block is not None:
                blocks.append(block)
    workbook.close()
    return blocks
