from app.parsers.normalizer import is_page_marker, normalize_text, repeated_line_keys


def test_hyphenated_word_is_joined():
    assert normalize_text("мощ-\nность 800 Вт") == "мощность 800 Вт"
    assert normalize_text("мощ-  \n  ность") == "мощность"


def test_real_hyphens_are_kept():
    assert normalize_text("Wi-Fi и Bluetooth") == "Wi-Fi и Bluetooth"
    # дефис в конце строки перед заглавной буквой или цифрой — не перенос
    assert "-\n800" in normalize_text("артикул BLD-\n800")


def test_soft_hyphen_is_removed():
    assert normalize_text("мощ­ность") == "мощность"


def test_ligatures_are_expanded():
    assert normalize_text("ﬁlter oﬀice ﬂow") == "filter office flow"


def test_extra_spaces_and_blank_lines_collapse():
    assert normalize_text("а   б в\t\tг\n\n\n\n\nд  ") == "а б в г\n\nд"


def test_repeated_footer_is_found_despite_page_number():
    pages = [
        ["Паспорт", "Текст первой страницы", "ООО «Дом» — стр. 1"],
        ["Комплект", "Текст второй страницы", "ООО «Дом» — стр. 2"],
    ]
    keys = repeated_line_keys(pages)
    assert keys == {"ооо «дом» — стр. #"}


def test_two_page_document_keeps_its_text():
    """«Большинство от двух — одна страница» не должно выметать обычные строки."""
    pages = [["Заголовок А", "Мощность 800 Вт"], ["Заголовок Б", "Мощность 900 Вт"]]
    assert repeated_line_keys(pages) == set()


def test_short_numeric_lines_are_not_headers():
    pages = [["Таблица", "6", "1"], ["Другая", "1", "6"]]
    assert repeated_line_keys(pages) == set()


def test_single_page_document_removes_nothing():
    assert repeated_line_keys([["строка", "строка"]]) == set()


def test_page_marker_detected():
    assert is_page_marker("ООО «ТехноДом» · site.example — стр. 1")
    assert is_page_marker("Страница 3 из 10")
    assert not is_page_marker("Количество скоростей: 6")
