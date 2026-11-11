"""Персональные данные: детекторы и маскирование до любого вызова модели и записи в лог.

Найденное заменяется метками «[ТЕЛЕФОН_1]»; исходные значения нигде не сохраняются: ни в
отчёте, ни в списке найденного — только вид и метка.
"""
import re
from dataclasses import dataclass

KIND_NAMES = {"phone": "телефон", "email": "email", "inn": "инн", "card": "карта"}
_LABELS = {"phone": "ТЕЛЕФОН", "email": "EMAIL", "inn": "ИНН", "card": "КАРТА"}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 13–19 цифр с пробелами или дефисами; настоящая карта проходит проверку Луна
_CARD_RE = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\w])"
)
# ровно 10 или 12 цифр, не часть артикула («BLD-1200») и не часть длинного числа
_INN_RE = re.compile(r"(?<![\w-])(?:\d{10}|\d{12})(?![\w-])")

_INN10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_N11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_N12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _checksum(digits: list[int], coefficients: tuple[int, ...]) -> int:
    return sum(d * c for d, c in zip(digits, coefficients, strict=False)) % 11 % 10


def _inn_checksum(value: str) -> bool:
    """ИНН с проверкой контрольной суммы.

    Одной длины мало: детектор «10 или 12 цифр» маскировал бы артикулы, суммы и даты
    и портил документ. У настоящего номера контрольные цифры сходятся.
    """
    if not value.isdigit():
        return False
    digits = [int(c) for c in value]
    if len(digits) == 10:
        return _checksum(digits, _INN10) == digits[9]
    if len(digits) == 12:
        return (
            _checksum(digits, _INN12_N11) == digits[10]
            and _checksum(digits, _INN12_N12) == digits[11]
        )
    return False


inn_valid = _inn_checksum


def luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class PiiFinding:
    """Найденное: вид и позиция. Само значение не хранится."""

    kind: str
    start: int
    end: int


def find_pii(text: str) -> list[PiiFinding]:
    """Все персональные данные в тексте без пересечений (раньше найденный вид приоритетнее)."""
    found: list[PiiFinding] = []

    def add(kind: str, match: re.Match) -> None:
        start, end = match.span()
        if all(end <= f.start or start >= f.end for f in found):
            found.append(PiiFinding(kind, start, end))

    for match in _EMAIL_RE.finditer(text):
        add("email", match)
    for match in _CARD_RE.finditer(text):
        if luhn_valid(match.group()):
            add("card", match)
    for match in _PHONE_RE.finditer(text):
        add("phone", match)
    for match in _INN_RE.finditer(text):
        if _inn_checksum(match.group()):
            add("inn", match)
    return sorted(found, key=lambda f: f.start)


def pii_kinds(text: str) -> list[str]:
    return sorted({KIND_NAMES[f.kind] for f in find_pii(text)})


class PiiMasker:
    """Маскирует тексты одной сборки контекста, метки стабильны: одно значение — одна метка."""

    def __init__(self) -> None:
        self._labels: dict[tuple[str, str], str] = {}
        self._numbers: dict[str, int] = {}
        self.counts: dict[str, int] = {}  # вид (по-русски) -> число вхождений

    @staticmethod
    def _identity(kind: str, value: str) -> str:
        return value.casefold() if kind == "email" else re.sub(r"\D", "", value)

    def _label(self, kind: str, value: str) -> str:
        key = (kind, self._identity(kind, value))
        if key not in self._labels:
            self._numbers[kind] = self._numbers.get(kind, 0) + 1
            self._labels[key] = f"[{_LABELS[kind]}_{self._numbers[kind]}]"
        return self._labels[key]

    def mask(self, text: str) -> str:
        findings = find_pii(text)
        # метки раздаём слева направо (первое найденное — «_1»), заменяем справа налево
        labels = [self._label(f.kind, text[f.start : f.end]) for f in findings]
        result = text
        for finding, label in reversed(list(zip(findings, labels, strict=True))):
            result = result[: finding.start] + label + result[finding.end :]
            name = KIND_NAMES[finding.kind]
            self.counts[name] = self.counts.get(name, 0) + 1
        return result


def mask_text(text: str) -> tuple[str, dict[str, int]]:
    """Замаскированный текст и сколько чего найдено (без исходных значений)."""
    masker = PiiMasker()
    masked = masker.mask(text)
    return masked, dict(masker.counts)


def contains_pii(text: str) -> bool:
    """Есть ли в тексте персональные данные (телефон, email, ИНН с верной суммой, карта)."""
    return bool(find_pii(text))


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Замаскированный текст и список найденного: вид и метка, без исходных значений."""
    masker = PiiMasker()
    findings = find_pii(text)
    masked = masker.mask(text)
    labels = re.findall(r"\[[A-ZА-ЯЁ]+_\d+\]", masked)
    result = []
    for finding, label in zip(findings, labels, strict=False):
        result.append({"kind": finding.kind, "label": label})
    return masked, result
