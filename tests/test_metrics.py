import asyncio

import pytest

from app.rag.context import build_context
from app.schemas import CardDraft, JudgeVerdict, SourceRef
from evals import judge as judge_module
from evals.judge import judge_prompt
from evals.metrics import (
    characteristic_matches,
    characteristics_accuracy,
    citation_report,
    judge_score,
    missing_fields_scores,
    norm_value,
    value_parts,
)


@pytest.mark.parametrize(
    "variant", ["800 Вт", "800 вт", "800 Вт.", "800Вт", "800 Вт", "  800   ВТ ", "800 Вт;"]
)
def test_equivalent_spellings_normalize_to_one_value(variant):
    assert norm_value(variant) == "800 вт"


def test_decimal_comma_and_thousands_separator():
    assert norm_value("1,5 л") == norm_value("1.5 л") == "1.5 л"
    assert norm_value("1 490 ₽") == norm_value("1490 ₽") == "1490 ₽"
    assert norm_value("2 190 ₽") == "2190 ₽"


def test_different_values_stay_different():
    assert norm_value("800 Вт") != norm_value("500 Вт")
    assert norm_value("1,5 л") != norm_value("1,2 кг")


def test_composite_value_is_split_but_decimal_comma_is_not():
    assert value_parts("КеттлПро 1.7, 2200 Вт, 1,7 л, сталь/пластик") == [
        "кеттлпро 1.7", "2200 вт", "1.7 л", "сталь/пластик",
    ]


def _card(characteristics=None, **kw) -> CardDraft:
    return CardDraft(title="Т", description="О", characteristics=characteristics or {}, **kw)


def test_characteristic_matches_after_normalization():
    card = _card({"мощность": "800 вт.", "Вес": "1,2 кг"})
    assert characteristic_matches("Мощность", "800 Вт", card)
    assert characteristic_matches("Вес", "1,2 кг", card)
    assert not characteristic_matches("Мощность", "900 Вт", card)
    assert not characteristic_matches("Цвет", "белый", card)


def test_more_detailed_card_value_still_matches():
    card = _card({"Питание": "220 В, 50 Гц, однофазное"})
    assert characteristic_matches("Питание", "220 В, 50 Гц", card)


def test_table_row_keyed_by_article_matches_by_parts():
    card = _card({"BLD-800": "МиксерПро 800, 800 Вт, графитовый, 2 190 ₽"})
    assert characteristic_matches("BLD-800", "МиксерПро 800, 800 Вт, графитовый, 2190 ₽", card)
    assert not characteristic_matches("BLD-800", "МиксерПро 800, 900 Вт, графитовый, 2190 ₽", card)


def test_accuracy_is_share_of_matched_expected_characteristics():
    card = _card({"Мощность": "800 Вт", "Вес": "1,2 кг"})
    expected = {"Мощность": "800 Вт", "Вес": "1,2 кг", "Цвет": "графитовый", "Модель": "X"}
    assert characteristics_accuracy(expected, card) == 0.5
    assert characteristics_accuracy({}, card) is None


def test_missing_fields_recall_and_precision():
    card = _card(missing_fields=["Цвет", "Гарантия", "Вес"])
    recall, precision = missing_fields_scores(["Цвет", "Гарантия", "Страна-производитель"], card)
    assert recall == pytest.approx(2 / 3)
    assert precision == pytest.approx(2 / 3)
    assert missing_fields_scores(["*"], _card()) == (None, 1.0)


# --- цитирование ---


def test_citation_ok_when_source_really_contains_the_value():
    card = _card(
        {"Мощность": "800 Вт"},
        sources=[SourceRef(field="Мощность", chunk_id="c1", quote="x", page=1)],
    )
    report = citation_report(card, {"c1": "Параметр\nМощность: 800 Вт\nВес: 1,2 кг"})
    assert report["card_ok"] and report["sources_valid"] == 1


def test_citation_catches_correct_value_taken_from_the_wrong_fragment():
    """Характеристика верная, но взята не оттуда, куда указывает ссылка."""
    card = _card(
        {"Мощность": "800 Вт"},
        sources=[SourceRef(field="Мощность", chunk_id="c2", quote="x")],
    )
    texts = {"c1": "Мощность: 800 Вт", 2: "Гарантия 24 месяца с даты продажи"}
    report = citation_report(card, texts)
    assert not report["card_ok"] and report["sources_valid"] == 0


def test_citation_fails_for_uncited_characteristic():
    card = _card({"Мощность": "800 Вт", "Вес": "1,2 кг"},
                 sources=[SourceRef(field="Мощность", chunk_id="c1", quote="x")])
    report = citation_report(card, {"c1": "Мощность: 800 Вт Вес: 1,2 кг"})
    assert not report["card_ok"] and report["uncited"] == ["Вес"]


def test_citation_probes_check_page_and_text():
    card = _card(
        {"Мощность": "800 Вт"},
        sources=[SourceRef(field="Мощность", chunk_id="c1", quote="x", page=1)],
    )
    probes = {"Мощность": {"page": 1, "text_contains": "800 Вт"}}
    assert citation_report(card, {"c1": "Мощность: 800 Вт"}, probes)["probes"] == "1/1"
    wrong_page = {"Мощность": {"page": 2, "text_contains": "800 Вт"}}
    assert citation_report(card, {"c1": "Мощность: 800 Вт"}, wrong_page)["probes"] == "0/1"


# --- судья ---


def test_judge_score_is_computed_by_code_from_named_keys():
    card = _card({"Мощность": "800 Вт", "Вес": "1,2 кг", "Цвет": "белый", "Модель": "X"})
    verdict = JudgeVerdict(unsupported_characteristics=["цвет", "Несуществующая"])
    assert judge_score(card, verdict) == 0.75  # лишнее имя от судьи не считается
    assert judge_score(_card(), verdict) is None


def test_judge_prompt_contains_fragments_and_card_but_not_hints():
    context = build_context(
        [
            {
                "chunk_id": "c7",
                "filename": "a.pdf",
                "text": "Мощность: 800 Вт",
                "metadata": {"page": 1, "section": "Характеристики"},
            }
        ]
    )
    card = _card({"Мощность": "800 Вт"}, confidence=0.9, missing_fields=["цвет"],
                 sources=[SourceRef(field="Мощность", chunk_id="c7", quote="x")])
    prompt = judge_prompt(context, card)
    assert "[c7 | a.pdf" in prompt and "Мощность: 800 Вт" in prompt  # фрагменты
    assert '"characteristics"' in prompt and '"title"' in prompt  # карточка
    assert "chunk_id" not in prompt and "confidence" not in prompt  # подсказок нет


def test_judge_returns_structured_verdict_from_substituted_model(monkeypatch):
    seen: list[tuple[str, str]] = []

    async def fake_run(agent, prompt):
        seen.append((agent.name, prompt))
        return '```json\n{"unsupported_characteristics": ["Цвет"], "comment": "цвета нет"}\n```'

    monkeypatch.setattr("app.services.structured.run_agent", fake_run)
    context = "[c7 | a.pdf | стр. 1 | С]\nМощность: 800 Вт"
    verdict = asyncio.run(judge_module.judge_card(context, _card({"Цвет": "белый"})))

    assert verdict.unsupported_characteristics == ["Цвет"]
    assert seen[0][0] == "Judge" and "Мощность: 800 Вт" in seen[0][1]


def test_citation_page_is_taken_from_database_not_from_model():
    sources = [SourceRef(field="Мощность", chunk_id="c1", quote="x")]
    card = _card({"Мощность": "800 Вт"}, sources=sources)
    probes = {"Мощность": {"page": 2, "text_contains": "800 Вт"}}
    report = citation_report(card, {"c1": "Мощность: 800 Вт"}, probes, chunk_pages={"c1": 2})
    assert report["probes"] == "1/1"  # у ссылки модели страницы нет, а во фрагменте она есть


def test_source_for_description_is_checked_by_quote():
    card = _card(
        {"Мощность": "800 Вт"},
        sources=[
            SourceRef(field="Мощность", chunk_id="c1", quote="x"),
            SourceRef(field="description", chunk_id="c1", quote="Металлическая ножка"),
        ],
    )
    text = "Мощность: 800 Вт. Металлическая ножка из нержавеющей стали."
    assert citation_report(card, {"c1": text})["card_ok"]
    invented = card.model_copy(update={"sources": [
        card.sources[0], SourceRef(field="description", chunk_id="c1", quote="Титановая ножка")]})
    assert citation_report(invented, {"c1": text})["sources_valid"] == 1
