import pytest

from app.services.json_utils import LLMJsonError, parse_llm_json

PAYLOAD = {"title": "Блендер", "confidence": 0.5}


def test_clean_json():
    assert parse_llm_json('{"title": "Блендер", "confidence": 0.5}') == PAYLOAD


def test_json_in_markdown_fences():
    text = '```json\n{"title": "Блендер", "confidence": 0.5}\n```'
    assert parse_llm_json(text) == PAYLOAD


def test_json_with_text_before_and_after():
    text = 'Конечно! Вот карточка:\n{"title": "Блендер", "confidence": 0.5}\nНадеюсь, помогло.'
    assert parse_llm_json(text) == PAYLOAD


def test_garbage_raises_clear_error():
    with pytest.raises(LLMJsonError, match="нет JSON"):
        parse_llm_json("Извините, я не могу ответить на этот вопрос.")


def test_empty_answer_raises():
    with pytest.raises(LLMJsonError, match="пустой"):
        parse_llm_json("   ")


def test_reasoning_block_is_ignored():
    text = '<think>{"a": 1} подумаю</think>\n{"title": "Блендер", "confidence": 0.5}'
    assert parse_llm_json(text) == PAYLOAD


def test_nested_object_is_returned_whole():
    text = 'Ответ: {"seo": {"keywords": ["a"]}, "title": "t"}'
    assert parse_llm_json(text) == {"seo": {"keywords": ["a"]}, "title": "t"}
