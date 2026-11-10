"""Строка таблицы → текст, который находится поиском по смыслу."""
import re

from app.parsers.models import Block

_KEY_HEADERS = ("артикул", "sku", "код")
_MAX_UNIT_LEN = 6


def cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def _split_header(header: str) -> tuple[str, str]:
    """«Мощность, Вт» → («мощность», «Вт»). Длинный хвост — не единица измерения."""
    name, sep, unit = header.rpartition(",")
    if sep and 0 < len(unit.strip()) <= _MAX_UNIT_LEN:
        return name.strip().lower(), unit.strip()
    return header.strip().lower(), ""


def _key_column(headers: list[str]) -> int:
    for index, header in enumerate(headers):
        if any(word in header.lower() for word in _KEY_HEADERS):
            return index
    return 0


def row_to_block(
    headers: list[str], values: list[str], *, page: int, section: str
) -> Block | None:
    """«BLD-800: мощность 800 Вт; чаша 1.5 л». Пустая строка таблицы → None.

    Заголовки колонок входят в текст строки, поэтому любая часть большой таблицы
    несёт их с собой: «800» без «мощность, Вт» ничего не значит.
    """
    cells = [cell_text(v) for v in values] + [""] * max(0, len(headers) - len(values))
    if not any(cells):
        return None
    key_index = _key_column(headers)
    key = cells[key_index]

    parts: list[str] = []
    brand: str | None = None
    for index, header in enumerate(headers):
        value = cells[index]
        if index == key_index or not value or not header:
            continue
        name, unit = _split_header(header)
        parts.append(" ".join(p for p in (name, value, unit) if p))
        if name == "бренд":
            brand = value

    body = "; ".join(parts)
    text = f"{key}: {body}" if key and body else (key or body)
    article = key if key else None
    return Block(kind="row", text=text, page=page, section=section, article=article, brand=brand)
