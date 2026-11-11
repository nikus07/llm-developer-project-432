"""Таблица → блок и строки таблицы → текст, который находится поиском по смыслу."""
import re

from app.parsers.models import Block

_KEY_HEADERS = ("артикул", "sku", "код")
_MAX_UNIT_LEN = 6
_KEY_VALUE_HEADERS = {"параметр", "характеристика", "свойство", "показатель"}


def cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def split_header(header: str) -> tuple[str, str]:
    """«Мощность, Вт» → («мощность», «Вт»). Длинный хвост — не единица измерения."""
    name, sep, unit = header.rpartition(",")
    if sep and 0 < len(unit.strip()) <= _MAX_UNIT_LEN:
        return name.strip().lower(), unit.strip()
    return header.strip().lower(), ""


def key_column(headers: list[str]) -> int | None:
    """Колонка артикула, если она есть: по ней строка таблицы получает метаданные."""
    for index, header in enumerate(headers):
        if any(word in header.lower() for word in _KEY_HEADERS):
            return index
    return None


def is_key_value(headers: list[str]) -> bool:
    """Двухколоночная таблица «параметр — значение»: строки читаются как «Мощность: 800 Вт»."""
    return len(headers) == 2 and headers[0].strip().lower() in _KEY_VALUE_HEADERS


def row_text(headers: list[str], values: list[str]) -> tuple[str, str | None, str | None]:
    """Строка таблицы → (текст, артикул, бренд). Пустая строка → ("", None, None).

    «BLD-800: мощность 800 Вт; чаша 1.5 л» — заголовки колонок входят в текст строки, поэтому
    любая часть большой таблицы несёт их с собой: «800» без «мощность, Вт» ничего не значит.
    """
    cells = [cell_text(v) for v in values] + [""] * max(0, len(headers) - len(values))
    if not any(cells):
        return "", None, None
    if is_key_value(headers):
        key, value = cells[0], cells[1]
        return (f"{key}: {value}" if key and value else (key or value)), None, None

    key_index = key_column(headers)
    key = cells[key_index] if key_index is not None else ""
    parts: list[str] = []
    brand: str | None = None
    for index, header in enumerate(headers):
        value = cells[index]
        if index == key_index or not value or not header:
            continue
        name, unit = split_header(header)
        parts.append(" ".join(p for p in (name, value, unit) if p))
        if name == "бренд":
            brand = value
    body = "; ".join(parts)
    text = f"{key}: {body}" if key and body else (key or body)
    return text, (key or None), brand


def table_block(rows: list[list[str]], *, page: int, section: str) -> Block | None:
    """Таблица из ячеек → блок вида «table» (первая строка — заголовки)."""
    rows = [[cell_text(c) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return None
    headers = rows[0]
    lines = [row_text(headers, row)[0] for row in rows[1:]]
    return Block(
        kind="table",
        content="\n".join([" | ".join(headers), *lines]),
        page=page,
        section=section,
        rows=rows,
    )
