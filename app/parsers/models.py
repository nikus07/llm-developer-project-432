from dataclasses import dataclass
from typing import Literal


class DocumentRejected(Exception):
    """Документ разобрать нельзя, и причину нужно показать пользователю."""


@dataclass
class Block:
    """Структурная единица документа: абзацы секции или одна строка таблицы."""

    kind: Literal["text", "row"]
    text: str
    page: int
    section: str
    article: str | None = None
    brand: str | None = None


@dataclass
class Chunk:
    idx: int
    kind: str
    text: str
    page: int
    section: str
    article: str | None = None
    brand: str | None = None
