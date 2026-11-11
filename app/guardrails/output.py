"""Фильтр выхода: карточка перед отдачей проверяется теми же детекторами.

Модель могла выполнить инъекцию частично или повторить контакт из документа; первые два
рубежа этого не увидят, поэтому смотрим на результат.
"""
import re
from dataclasses import dataclass, field

from app.guardrails.injection import scan_rules
from app.guardrails.pii import PiiMasker, pii_kinds
from app.schemas import Card

_WORDS_RE = re.compile(r"\w+", re.UNICODE)
_SHINGLE = 4  # столько слов подряд, повторённых из исключённого фрагмента, — след инъекции


@dataclass
class OutputCheck:
    card: Card
    findings: list[dict] = field(default_factory=list)  # вид и место, без самих значений

    @property
    def clean(self) -> bool:
        return not self.findings


def _walk(value, fn):
    """Применяет fn ко всем строкам вложенной структуры."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk(v, fn) for v in value]
    if isinstance(value, dict):
        return {_walk(k, fn): _walk(v, fn) for k, v in value.items()}
    return value


def _strings(value, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, path)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key, path)
            yield from _strings(item, f"{path}.{key}" if path else key)


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = [w.casefold() for w in _WORDS_RE.findall(text)]
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def check_output(card: Card, excluded_texts: list[str] | None = None) -> OutputCheck:
    """Контакты, следы служебных инструкций и повторы исключённых фрагментов.

    Контакты в карточке заменяются метками, остальное только отмечается: решает человек.
    """
    data = card.model_dump()
    findings: list[dict] = []
    excluded = set()
    for text in excluded_texts or []:
        excluded |= _shingles(text)

    for path, text in _strings(data):
        field_name = path.split(".")[0] or "card"
        for kind in pii_kinds(text):
            findings.append({"kind": f"персональные данные: {kind}", "field": field_name})
        for rule in scan_rules(text):
            findings.append({"kind": f"след инструкции: {rule}", "field": field_name})
        if excluded and _shingles(text) & excluded:
            findings.append({"kind": "повтор исключённого фрагмента", "field": field_name})

    masker = PiiMasker()
    cleaned = Card.model_validate(_walk(data, masker.mask))
    return OutputCheck(card=cleaned, findings=findings)
