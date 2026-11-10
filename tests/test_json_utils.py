import pytest

from app.services.json_utils import JsonParseError, extract_json

PAYLOAD = {"title": "Блендер", "confidence": 0.5}


def test_clean_json():
    assert extract_json('{"title": "Блендер", "confidence": 0.5}') == PAYLOAD


def test_json_in_markdown_fences():
    text = '```json\n{"title": "Блендер", "confidence": 0.5}\n```'
    assert extract_json(text) == PAYLOAD


def test_json_with_text_before_and_after():
    text = 'Конечно! Вот карточка:\n{"title": "Блендер", "confidence": 0.5}\nНадеюсь, помогло.'
    assert extract_json(text) == PAYLOAD


def test_garbage_raises_clear_error():
    with pytest.raises(JsonParseError, match="нет JSON"):
        extract_json("Извините, я не могу ответить на этот вопрос.")


def test_empty_answer_raises():
    with pytest.raises(JsonParseError, match="пустой"):
        extract_json("   ")


def test_reasoning_block_is_ignored():
    text = '<think>{"a": 1} подумаю</think>\n{"title": "Блендер", "confidence": 0.5}'
    assert extract_json(text) == PAYLOAD


def test_nested_object_is_returned_whole():
    text = 'Ответ: {"seo": {"keywords": ["a"]}, "title": "t"}'
    assert extract_json(text) == {"seo": {"keywords": ["a"]}, "title": "t"}
