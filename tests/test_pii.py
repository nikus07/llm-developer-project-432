import pytest

from app.guardrails.pii import PiiMasker, find_pii, inn_valid, luhn_valid, mask_text, pii_kinds


@pytest.mark.parametrize(
    "phone",
    [
        "+7 926 555-14-08",
        "+7(926)555-14-08",
        "8 926 555 14 08",
        "8-926-555-14-08",
        "+79265551408",
        "8 (800) 100-22-47",
    ],
)
def test_phone_is_masked(phone):
    masked, counts = mask_text(f"Звоните: {phone}, спросите Анну.")
    assert phone not in masked
    assert "[ТЕЛЕФОН_1]" in masked
    assert counts == {"телефон": 1}


def test_email_is_masked():
    masked, counts = mask_text("Пишите на a.smirnova@technodom.example или в чат")
    assert "technodom" not in masked and "[EMAIL_1]" in masked
    assert counts == {"email": 1}


def test_inn_with_valid_checksum_is_masked():
    # 7712345671 — ИНН организации с верной контрольной цифрой
    assert inn_valid("7712345671")
    masked, counts = mask_text("ИНН компании: 7712345671")
    assert masked == "ИНН компании: [ИНН_1]" and counts == {"инн": 1}


def test_inn_of_individual_with_two_check_digits():
    assert inn_valid("500100732259")  # 12 цифр: проверяются обе контрольные
    assert not inn_valid("500100732258")


def test_number_with_wrong_checksum_is_left_alone():
    """Одна длина — не признак ИНН: артикулы, суммы и даты маскировать нельзя."""
    for number in ("1234567890", "7712345670", "123456789012"):
        assert not inn_valid(number)
        text = f"Партия {number} шт."
        assert mask_text(text) == (text, {})


def test_ordinary_numbers_are_not_touched():
    text = (
        "Артикул BLD-1200, цена 3 450 ₽, вес 1,2 кг, 2200 Вт, 220 В, 50 Гц; "
        "ОГРН: 1157746123450; дата 12.03.2026; код 20260311"
    )
    assert mask_text(text) == (text, {})


def test_card_number_needs_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    masked, counts = mask_text("Оплата картой 4111 1111 1111 1111 прошла")
    assert "4111" not in masked and "[КАРТА_1]" in masked and counts == {"карта": 1}
    wrong = "номер 4111 1111 1111 1112 — не карта"
    assert mask_text(wrong) == (wrong, {})


def test_same_value_gets_same_label_and_labels_count_from_one():
    masker = PiiMasker()
    first = masker.mask("Тел. +7 926 555-14-08 и +7 495 085-21-62")
    second = masker.mask("Опять +7 926 555-14-08")
    assert first == "Тел. [ТЕЛЕФОН_1] и [ТЕЛЕФОН_2]"
    assert second == "Опять [ТЕЛЕФОН_1]"  # метка стабильна между фрагментами
    assert masker.counts == {"телефон": 3}


def test_findings_never_contain_original_values():
    findings = find_pii("Звоните +7 926 555-14-08, a@b.example")
    assert {f.kind for f in findings} == {"phone", "email"}
    assert all(not hasattr(f, "value") for f in findings)
    _, counts = mask_text("+7 926 555-14-08")
    assert counts == {"телефон": 1}  # только вид и число


def test_pii_kinds_lists_kinds_in_russian():
    assert pii_kinds("+7 926 555-14-08 a@b.example") == ["email", "телефон"]
    assert pii_kinds("ничего личного") == []


def test_manager_block_from_commercial_offer():
    text = (
        "Смирнова Анна Викторовна, менеджер\nТелефон: +7 926 555-14-08\n"
        "Email: a.smirnova@technodom.example\nИНН компании: 7712345671\nОГРН: 1157746123450"
    )
    masked, counts = mask_text(text)
    for raw in ("926 555-14-08", "smirnova@", "7712345671"):
        assert raw not in masked
    assert "1157746123450" in masked  # ОГРН остаётся
    assert counts == {"телефон": 1, "email": 1, "инн": 1}
