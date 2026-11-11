"""Лечение невалидных ответов и точечная переделка: модель подменена."""
import asyncio
import json

import pytest

from app.agents.prompts import generator_agent
from app.core.config import settings
from app.schemas import CardDraft
from app.services import structured
from app.services.structured import StructuredOutputError, call_structured

GOOD = {
    "title": "МиксерПро 800",
    "description": "Погружной блендер на 800 Вт с металлической ножкой.",
    "characteristics": {"Мощность": "800 Вт"},
    "confidence": 0.8,
}


@pytest.fixture
def fake_llm(monkeypatch):
    calls: list[tuple[str, str]] = []
    scenario: list = []

    async def fake_run(agent, prompt):
        calls.append((agent.name, prompt))
        return scenario[len(calls) - 1]

    monkeypatch.setattr(structured, "run_agent", fake_run)
    return calls, scenario


def _run(**kw):
    return asyncio.run(call_structured(generator_agent(), "факты", CardDraft, **kw))


def test_invalid_json_triggers_retry_with_error_in_prompt(fake_llm):
    calls, scenario = fake_llm
    scenario += ["это не JSON, извините", json.dumps(GOOD, ensure_ascii=False)]

    card = _run()

    assert card.title == "МиксерПро 800"
    assert len(calls) == 2
    retry_prompt = calls[1][1]
    assert "не прошёл проверку" in retry_prompt
    assert "нет JSON-объекта" in retry_prompt  # текст ошибки ушёл в модель
    assert "это не JSON" in retry_prompt  # и сам плохой ответ тоже


def test_empty_answer_gives_error(fake_llm):
    calls, scenario = fake_llm
    scenario += ["", "", ""]

    with pytest.raises(StructuredOutputError, match="пустой ответ"):
        _run()
    assert len(calls) == 1 + settings.structured_max_repairs  # попыток ограниченное число


def test_repairs_are_limited(fake_llm):
    calls, scenario = fake_llm
    scenario += ["мусор"] * 10
    with pytest.raises(StructuredOutputError):
        _run(max_repairs=1)
    assert len(calls) == 2


def test_long_title_is_fixed_pointwise(fake_llm):
    calls, scenario = fake_llm
    long_title = "МиксерПро 800 " * 10
    broken = {**GOOD, "title": long_title}
    scenario += [
        json.dumps(broken, ensure_ascii=False),
        '{"title": "МиксерПро 800 — блендер"}',  # ответ правщика одного поля
    ]

    card = _run(field_repair=True)

    assert card.title == "МиксерПро 800 — блендер"
    assert card.description == GOOD["description"]  # остальное не переписывалось
    assert [name for name, _ in calls] == ["Generator", "FieldFixer"]  # генератор — один раз
    assert "«title»" in calls[1][1]
    assert long_title.strip() in calls[1][1]  # правщик получил текущий черновик


def test_without_field_repair_whole_answer_is_regenerated(fake_llm):
    calls, scenario = fake_llm
    scenario += [
        json.dumps({**GOOD, "title": "а" * 100}, ensure_ascii=False),
        json.dumps(GOOD, ensure_ascii=False),
    ]
    _run(field_repair=False)
    assert [name for name, _ in calls] == ["Generator", "Generator"]


def test_valid_model_instance_is_returned_as_is(fake_llm):
    calls, scenario = fake_llm
    scenario.append(CardDraft(**GOOD))
    assert _run().title == "МиксерПро 800"
    assert len(calls) == 1


def test_unchanged_field_fix_is_reported_to_the_model(fake_llm):
    calls, scenario = fake_llm
    long_title = "МиксерПро 800 " * 10
    scenario += [
        json.dumps({**GOOD, "title": long_title}, ensure_ascii=False),
        json.dumps({"title": long_title}, ensure_ascii=False),  # правщик ничего не изменил
        '{"title": "МиксерПро 800"}',
    ]

    card = _run(field_repair=True)

    assert card.title == "МиксерПро 800"
    assert "то же значение" in calls[2][1]
