"""xlsx: спецификация «артикул → параметры», заголовки колонок — первая строка."""
from io import BytesIO

from openpyxl import load_workbook

from app.parsers.models import Block, DocumentRejected
from app.parsers.table import table_block


def parse_xlsx(data: bytes) -> list[Block]:
    try:
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise DocumentRejected(f"не удалось открыть xlsx: {exc}") from exc

    blocks: list[Block] = []
    for sheet_number, sheet in enumerate(workbook.worksheets, start=1):
        rows = [list(row) for row in sheet.iter_rows(values_only=True)]
        block = table_block(rows, page=sheet_number, section=sheet.title)
        if block is not None:
            blocks.append(block)
    workbook.close()
    return blocks
