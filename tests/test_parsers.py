"""Разбор и резка на выданных файлах из data/."""
from pathlib import Path

import pytest

from app.parsers import docx, pdf, xlsx
from app.parsers.chunker import chunk_blocks, split_text
from app.parsers.models import DocumentRejected

DATA = Path(__file__).resolve().parent.parent / "data"


def _text(chunks) -> str:
    return "\n".join(c.text for c in chunks)


def test_pdf_passport_has_expected_lines_and_sections():
    chunks = chunk_blocks(pdf.parse_pdf(DATA / "blender_passport.pdf"))
    text = _text(chunks)
    assert "Артикул: BLD-800" in text
    assert "Мощность: 800 Вт" in text
    assert {c.section for c in chunks} >= {"Технические характеристики", "Комплект поставки"}
    assert {c.page for c in chunks} == {1, 2}


def test_pdf_footer_is_removed_on_two_pages_and_on_one_page():
    for name in ("blender_passport.pdf", "kettle_manual.pdf"):
        text = _text(chunk_blocks(pdf.parse_pdf(DATA / name)))
        assert "стр." not in text
        assert "supplies@technodom.example" not in text


def test_pdf_two_page_document_keeps_text_of_both_pages():
    chunks = chunk_blocks(pdf.parse_pdf(DATA / "blender_passport.pdf"))
    assert "Гарантия 24 месяца с даты продажи" in _text(chunks)  # страница 2
    assert "Металлическая ножка" in _text(chunks)  # страница 1


def test_scanned_pdf_is_rejected_with_reason():
    with pytest.raises(DocumentRejected, match="нет текстового слоя"):
        pdf.parse_pdf(DATA / "boiler_scan.pdf")


def test_docx_sections_and_price_table():
    chunks = chunk_blocks(docx.parse_docx(DATA / "blender_kp.docx"))
    rows = [c for c in chunks if c.kind == "row"]
    assert rows and rows[0].section == "Прайс-лист (опт)"
    assert "BLD-800: модель МиксерПро 800; мощность 800 Вт" in rows[0].text
    assert "BLD-800" in rows[0].article
    assert "Условия" in {c.section for c in chunks}


def test_xlsx_rows_read_as_article_parameter_value():
    chunks = chunk_blocks(xlsx.parse_xlsx(DATA / "kettle_spec.xlsx"))
    text = _text(chunks)
    assert "KTL-1700: бренд КеттлПро; модель 1.7; мощность 2200 Вт; объём 1,7 л" in text
    assert all(c.brand == "КеттлПро" for c in chunks)
    assert all(c.section == "Спецификация" and c.page == 1 for c in chunks)


def test_table_rows_are_never_torn_between_chunks():
    blocks = xlsx.parse_xlsx(DATA / "kettle_spec.xlsx")
    expected_rows = {b.text for b in blocks}
    chunks = chunk_blocks(blocks, size=150)  # заведомо меньше трёх строк вместе
    assert len(chunks) > 1
    found = [line for c in chunks for line in c.text.split("\n")]
    assert set(found) == expected_rows
    assert len(found) == len(blocks)  # ни одна строка не продублирована и не разорвана


def test_every_chunk_has_document_metadata():
    for parse, name in ((pdf.parse_pdf, "blender_passport.pdf"),
                        (docx.parse_docx, "blender_kp.docx"),
                        (xlsx.parse_xlsx, "kettle_spec.xlsx")):
        for chunk in chunk_blocks(parse(DATA / name)):
            assert chunk.page >= 1
            assert chunk.section


def test_text_is_split_with_overlap_on_line_boundaries():
    lines = [f"Строка номер {i} с некоторым текстом" for i in range(30)]
    pieces = split_text("\n".join(lines), size=200, overlap=70)
    assert len(pieces) > 2
    for before, after in zip(pieces, pieces[1:], strict=False):
        tail = before.split("\n")[-1]
        assert tail in after.split("\n")  # хвост предыдущего повторён в следующем
    assert all(len(p) <= 200 + 40 for p in pieces)
    assert {line for p in pieces for line in p.split("\n")} == set(lines)  # ничего не потеряно


def test_long_line_is_split_by_words():
    pieces = split_text("слово " * 100, size=100, overlap=0)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces).split() == ["слово"] * 100
