"""Нормализация текста документа: колонтитулы, переносы, лигатуры, пробелы."""
import math
import re
from dataclasses import replace

from app.parsers.models import Block

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE_RE = re.compile("|".join(_LIGATURES))
_SPACES_RE = re.compile(r"[ \t   ]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# перенос: буква, дефис в конце строки, продолжение слова с маленькой буквы на следующей
_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-zа-яё])")
_SOFT_HYPHEN_RE = re.compile("­[ \t]*\n?[ \t]*")
_PAGE_MARKER_RE = re.compile(
    r"(?:стр\.|страница|page|p\.)\s*\d+(?:\s*(?:из|of|/)\s*\d+)?\s*$", re.IGNORECASE
)

# колонтитул стоит у края страницы: смотрим на столько строк сверху и снизу
_EDGE_LINES = 2
_MIN_HEADER_LEN = 4


def clean_text(text: str) -> str:
    """Лигатуры, переносы через дефис, лишние пробелы и пустые строки."""
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SOFT_HYPHEN_RE.sub("", text)
    # «мощ-\nность» → «мощность»; «Wi-Fi» не трогаем: дефис должен стоять в конце строки
    text = _HYPHEN_RE.sub("", text)
    text = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


_PAGE_NUMBER_RE = re.compile(
    r"(стр\.|страница|page|p\.)\s*\d+(?:\s*(?:из|of|/)\s*\d+)?", re.IGNORECASE
)


def line_key(line: str) -> str:
    """Ключ для сравнения строк между страницами.

    Меняется только номер страницы («стр. 1» и «стр. 2» — один ключ). Остальные цифры
    остаются: «мощность 800 Вт» и «мощность 900 Вт» — разные строки.
    """
    key = _SPACES_RE.sub(" ", line).strip().lower()
    return _PAGE_NUMBER_RE.sub(lambda m: m.group(1) + " #", key)


def is_page_marker(line: str) -> bool:
    """«… — стр. 3» в конце строки: номер страницы выдаёт колонтитул даже в одностраничном pdf."""
    return bool(_PAGE_MARKER_RE.search(line.strip()))


def edge_lines(lines: list[str]) -> list[str]:
    filled = [line for line in lines if line.strip()]
    return filled[:_EDGE_LINES] + filled[-_EDGE_LINES:]


def repeated_line_keys(pages: list[list[str]]) -> set[str]:
    """Колонтитулы: строки у края страницы, которые повторяются на страницах.

    Порог — не меньше двух страниц И не меньше половины страниц документа. Правило
    «на большинстве страниц» на двух страницах давало бы «одна из двух», то есть любую
    строку любой страницы, и выметало бы весь текст. На одной странице колонтитул
    от обычной строки отличить нельзя, поэтому там ничего не удаляется.
    """
    if len(pages) < 2:
        return set()
    threshold = max(2, math.ceil(len(pages) / 2))
    seen: dict[str, int] = {}
    for lines in pages:
        keys = {line_key(line) for line in edge_lines(lines)}
        for key in keys:
            if len(key) >= _MIN_HEADER_LEN:
                seen[key] = seen.get(key, 0) + 1
    return {key for key, count in seen.items() if count >= threshold}


def normalize(blocks: list[Block]) -> list[Block]:
    """Нормализация блоков документа: колонтитулы, переносы, лигатуры, пробелы.

    Колонтитул виден только на фоне других страниц, поэтому его убирают здесь, а не при
    разборе: строки у края страницы, повторяющиеся на страницах, и строки с номером страницы.
    Таблицы остаются как есть.
    """
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        if block.kind == "text":
            by_page.setdefault(block.page, []).extend(block.content.split("\n"))
    lines_by_page = [by_page[page] for page in sorted(by_page)]
    repeated = repeated_line_keys(lines_by_page)
    for lines in lines_by_page:
        repeated |= {line_key(ln) for ln in edge_lines(lines) if is_page_marker(ln)}

    result: list[Block] = []
    for block in blocks:
        if block.kind != "text":
            result.append(block)
            continue
        kept = [ln for ln in block.content.split("\n") if line_key(ln) not in repeated]
        text = clean_text("\n".join(kept))
        if text:
            result.append(replace(block, content=text))
    return result
