from dataclasses import dataclass, field
from typing import Literal


class DocumentRejected(Exception):
    """Документ разобрать нельзя, и причину нужно показать пользователю."""


@dataclass
class Block:
    """Структурная единица документа: абзацы секции или таблица.

    Для таблицы `rows` — ячейки по строкам (первая строка — заголовки колонок), а
    `content` — те же строки текстом, по одной на строку таблицы.
    """

    kind: Literal["text", "table"]
    content: str
    page: int
    section: str
    rows: list[list[str]] = field(default_factory=list)
