import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.schemas import CardDraft


def _card(**kw) -> dict:
    return {"title": "Блендер", "description": "Описание.", **kw}


def test_title_longer_than_limit_is_rejected():
    with pytest.raises(ValidationError, match="сократи"):
        CardDraft(**_card(title="а" * (settings.card_title_max_len + 1)))


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        CardDraft(**_card(confidence=1.5))
    with pytest.raises(ValidationError):
        CardDraft(**_card(confidence=-0.1))
    assert CardDraft(**_card(confidence=0.0)).confidence == 0.0


def test_empty_characteristics_are_dropped():
    card = CardDraft(**_card(characteristics={"Мощность": "800 Вт", "Цвет": "", "Вес": "  "}))
    assert card.characteristics == {"Мощность": "800 Вт"}


def test_field_cannot_be_in_both_lists():
    with pytest.raises(ValidationError, match="одновременно"):
        CardDraft(**_card(characteristics={"Цвет": "красный"}, missing_fields=["Цвет"]))


def test_nested_seo_and_sources_are_parsed():
    card = CardDraft(
        **_card(
            seo={"meta_title": "t", "keywords": ["блендер"]},
            sources=[{"field": "Мощность", "quote": "Мощность 800 Вт"}],
        )
    )
    assert card.seo.keywords == ["блендер"]
    assert card.sources[0].quote == "Мощность 800 Вт"
