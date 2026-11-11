"""Контекст генерации: найденные фрагменты, оформленные как опознаваемые источники."""
import re
from dataclasses import asdict, dataclass, field
from html import escape

from app.core.config import settings

_SPACES_RE = re.compile(r"\s+")


@dataclass
class ContextItem:
    chunk_id: int
    document: str
    page: int
    section: str
    text: str

    def render(self) -> str:
        """Источник для промпта: id — то, на что модель ссылается, а приложение проверяет.

        Атрибуты экранируются: имя файла или заголовок секции от постороннего человека не
        должны закрыть тег и вписать в промпт собственные инструкции.
        """
        attrs = (
            f'id="{self.chunk_id}" document="{escape(self.document, quote=True)}" '
            f'page="{self.page}" section="{escape(self.section, quote=True)}"'
        )
        return f"<source {attrs}>\n{self.text}\n</source>"


@dataclass
class Context:
    items: list[ContextItem] = field(default_factory=list)
    duplicates_removed: int = 0
    omitted: int = 0  # не поместились в предел размера
    pii_masked: dict[str, int] = field(default_factory=dict)  # вид -> сколько замаскировано

    @property
    def chunk_ids(self) -> set[int]:
        return {item.chunk_id for item in self.items}

    @property
    def text(self) -> str:
        return "\n\n".join(item.render() for item in self.items)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        return cls(
            items=[ContextItem(**item) for item in data["items"]],
            duplicates_removed=data.get("duplicates_removed", 0),
            omitted=data.get("omitted", 0),
            pii_masked=data.get("pii_masked", {}),
        )


def context_chunk_ids(context: Context) -> set[int]:
    """Идентификаторы фрагментов, которые были в контексте, поданном модели."""
    return context.chunk_ids


def _fingerprint(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip().casefold()


def build_context(hits: list[dict], max_chars: int | None = None) -> Context:
    """Собирает контекст из найденных фрагментов, в порядке релевантности.

    - одинаковый текст из разных документов (или тот же фрагмент, найденный дважды)
      берётся один раз: иначе половина контекста уйдёт на повторы;
    - размер ограничен: фрагмент, который не помещается, пропускается (меньшие, что идут
      дальше, ещё могут поместиться).
    """
    limit = max_chars or settings.context_max_chars
    context = Context()
    seen: set[str] = set()
    used = 0
    for hit in hits:
        fingerprint = _fingerprint(hit["text"])
        if fingerprint in seen:
            context.duplicates_removed += 1
            continue
        item = ContextItem(
            chunk_id=hit["id"],
            document=hit["document"],
            page=hit["page"],
            section=hit["section"],
            text=hit["text"],
        )
        size = len(item.render()) + 2  # разделитель между источниками
        if used + size > limit:
            context.omitted += 1
            continue
        seen.add(fingerprint)
        context.items.append(item)
        used += size
    return context
