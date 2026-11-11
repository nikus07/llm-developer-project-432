"""Метрики качества генерации: чистые функции, без модели и без базы."""
import re

from app.schemas import Card, JudgeVerdict

_SPACE_CHARS_RE = re.compile("[   ]")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_THOUSANDS_RE = re.compile(r"(?<=\d) (?=\d{3}(?!\d))")
_NUMBER_UNIT_RE = re.compile(r"(?<=\d)(?=[^\W\d_])")
_PARTS_RE = re.compile(r",\s+|;\s*")


def norm_value(value: str) -> str:
    """Значения сравниваются не буквально: «800 Вт», «800 вт» и «800 Вт.» — одно и то же.

    Регистр, пробелы (в том числе неразрывные), десятичная запятая («1,5» = «1.5»),
    разделитель тысяч («1 490» = «1490»), слитная единица («800Вт» = «800 вт»),
    кавычки и точка в конце.
    """
    text = _SPACE_CHARS_RE.sub(" ", str(value)).casefold()
    text = text.replace("«", "").replace("»", "").replace('"', "")
    text = _DECIMAL_COMMA_RE.sub(".", text)
    text = _THOUSANDS_RE.sub("", text)
    text = _NUMBER_UNIT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".;,: ").strip()


def norm_key(key: str) -> str:
    return re.sub(r"\s+", " ", key).strip().casefold().rstrip(":")


def value_parts(value: str) -> list[str]:
    """Составное значение «a, b, c» → части; запятая внутри числа («1,7 л») не делит."""
    return [p for p in (norm_value(part) for part in _PARTS_RE.split(value)) if p]


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and needle in haystack


def characteristic_matches(key: str, expected: str, card: Card) -> bool:
    """Совпала ли ожидаемая характеристика с карточкой.

    Ключ есть в карточке — значения сравниваются после нормализации (равенство или
    ожидаемое целиком внутри более подробного значения карточки). Ключа нет (у таблиц ключ —
    артикул, а значение составное) — артикул и все части значения должны найтись в
    характеристиках карточки.
    """
    expected_n = norm_value(expected)
    actual = {norm_key(k): norm_value(v) for k, v in card.characteristics.items()}
    key_n = norm_key(key)
    if key_n in actual:
        return actual[key_n] == expected_n or _contains(actual[key_n], expected_n)

    haystack = " ; ".join(f"{k} {v}" for k, v in actual.items())
    return _contains(haystack, key_n) and all(
        _contains(haystack, part) for part in value_parts(expected)
    )


def characteristics_accuracy(expected: dict[str, str], card: Card) -> float | None:
    """Метрика 1: доля ожидаемых характеристик, совпавших с эталоном."""
    if not expected:
        return None
    hits = sum(characteristic_matches(k, v, card) for k, v in expected.items())
    return hits / len(expected)


def missing_fields_scores(expected: list[str], card: Card) -> tuple[float | None, float | None]:
    """(полнота, точность) списка недостающих полей относительно эталона."""
    want = {norm_key(f) for f in expected if f != "*"}
    got = {norm_key(f) for f in card.missing_fields}
    hit = len(want & got)
    recall = hit / len(want) if want else None
    precision = hit / len(got) if got else (1.0 if not want else None)
    return recall, precision


def _value_in_text(value: str, text: str) -> bool:
    text_n = norm_value(text)
    parts = value_parts(value)
    return bool(parts) and all(_contains(text_n, p) for p in parts)


def citation_report(
    card: Card,
    chunk_texts: dict[int, str],
    probes: dict | None = None,
    chunk_pages: dict[int, int] | None = None,
) -> dict:
    """Метрика 2: заявленный источник действительно содержит то, на что ссылается.

    Ловит случай «характеристика верная, но взята не оттуда»: значение должно найтись в
    тексте именно того фрагмента, который указан в ссылке. Ссылка не на характеристику
    (на описание, заголовок) проверяется по цитате: она должна дословно быть во фрагменте.
    Страницу берём из базы по chunk_id: модель её не возвращает.
    """
    values = {norm_key(k): v for k, v in card.characteristics.items()}
    valid = 0
    for source in card.sources:
        text = chunk_texts.get(source.chunk_id) if source.chunk_id is not None else None
        value = values.get(norm_key(source.field))
        claim = value if value is not None else source.quote
        if text is not None and claim and _value_in_text(claim, text):
            valid += 1
    cited = {norm_key(s.field) for s in card.sources}
    uncited = [k for k in card.characteristics if norm_key(k) not in cited]
    total = len(card.sources)

    probe_hits = probe_total = 0
    for field, probe in (probes or {}).items():
        probe_total += 1
        for source in card.sources:
            if norm_key(source.field) != norm_key(field):
                continue
            text = chunk_texts.get(source.chunk_id) if source.chunk_id is not None else None
            page = (chunk_pages or {}).get(source.chunk_id, source.page)
            page_ok = probe.get("page") is None or page == probe["page"]
            if text is not None and page_ok and _contains(
                norm_value(text), norm_value(probe["text_contains"])
            ):
                probe_hits += 1
                break
    return {
        "card_ok": total > 0 and valid == total and not uncited,
        "sources_valid": valid,
        "sources_total": total,
        "uncited": uncited,
        "probes": f"{probe_hits}/{probe_total}" if probe_total else None,
    }


def judge_score(card: Card, verdict: JudgeVerdict) -> float | None:
    """Метрика 3: доля характеристик, которые судья счёл подтверждёнными источниками.

    Судья называет неподтверждённые, а считает код: модели не доверяем арифметику.
    Это оценка, а не истина.
    """
    keys = {norm_key(k) for k in card.characteristics}
    if not keys:
        return None
    unsupported = {norm_key(k) for k in verdict.unsupported_characteristics} & keys
    return 1 - len(unsupported) / len(keys)
