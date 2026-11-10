"""Разбор грязного ответа модели в JSON-объект."""
import json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class JsonParseError(ValueError):
    """Из ответа нельзя достать JSON-объект."""


def _loads_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели.

    Переживает: чистый JSON, обёртку из markdown-заборов, текст до и после JSON,
    блок рассуждений <think>. Иначе — JsonParseError с понятным текстом.
    """
    text = _THINK_RE.sub("", text or "").strip()
    if not text:
        raise JsonParseError("пустой ответ модели")

    if (data := _loads_object(text)) is not None:
        return data

    for match in _FENCE_RE.finditer(text):
        if (data := _loads_object(match.group(1))) is not None:
            return data

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, index)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value

    raise JsonParseError(f"в ответе нет JSON-объекта: {text[:80]!r}")
