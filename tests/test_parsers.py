"""Разбор и резка на выданных файлах из data/."""
from pathlib import Path

import pytest

from app.parsers import docx, pdf, xlsx
from app.parsers.chunker import chunk_blocks, split_text
from app.parsers.models import DocumentRejected
from app.parsers.normalizer import normalize

DATA = Path(__file__).resolve().parent.parent / "data"


def _bytes(name: str) -> bytes:
    return (DATA / name).read_bytes()


def _text(chunks) -> str:
    return "\n".join(c["text"] for c in chunks)


def test_pdf_passport_has_expected_lines_and_sections():
    chunks = chunk_blocks(pdf.parse_pdf(_bytes("blender_passport.pdf")), "d1")
    text = _text(chunks)
    assert "Артикул: BLD-800" in text
    assert "Мощность: 800 Вт" in text
    sections = {c["metadata"]["section"] for c in chunks}
    assert sections >= {"Технические характеристики", "Комплект поставки"}
    assert {c["metadata"]["page"] for c in chunks} == {1, 2}


def test_pdf_footer_is_removed_on_two_pages_and_on_one_page():
    for name in ("blender_passport.pdf", "kettle_manual.pdf"):
        text = _text(chunk_blocks(normalize(pdf.parse_pdf(_bytes(name))), "d1"))
        assert "стр." not in text
        assert "supplies@technodom.example" not in text


def test_pdf_two_page_document_keeps_text_of_both_pages():
    chunks = chunk_blocks(pdf.parse_pdf(_bytes("blender_passport.pdf")), "d1")
    assert "Гарантия 24 месяца с даты продажи" in _text(chunks)  # страница 2
    assert "Металлическая ножка" in _text(chunks)  # страница 1


def test_scanned_pdf_is_rejected_with_reason():
    with pytest.raises(DocumentRejected, match="нет текстового слоя"):
        pdf.parse_pdf(_bytes("boiler_scan.pdf"))


def test_docx_sections_and_price_table():
    chunks = chunk_blocks(docx.parse_docx(_bytes("blender_kp.docx")), "d1")
    rows = [c for c in chunks if c["metadata"]["kind"] == "row"]
    assert rows and rows[0]["metadata"]["section"] == "Прайс-лист (опт)"
    (row,) = [c for c in rows if c["metadata"].get("articul") == "BLD-800"]
    assert "BLD-800: модель МиксерПро 800; мощность 800 Вт" in row["text"]
    assert "Условия" in {c["metadata"]["section"] for c in chunks}


def test_xlsx_rows_read_as_article_parameter_value():
    chunks = chunk_blocks(xlsx.parse_xlsx(_bytes("kettle_spec.xlsx")), "d1")
    text = _text(chunks)
    assert "KTL-1700: бренд КеттлПро; модель 1.7; мощность 2200 Вт; объём 1,7 л" in text
    assert all(c["metadata"]["brand"] == "КеттлПро" for c in chunks)
    assert all(
        c["metadata"]["section"] == "Спецификация" and c["metadata"]["page"] == 1 for c in chunks
    )


def test_table_rows_are_never_torn_between_chunks():
    blocks = xlsx.parse_xlsx(_bytes("kettle_spec.xlsx"))
    expected_rows = {line for b in blocks for line in b.content.split("\n")[1:]}  # без шапки
    chunks = chunk_blocks(blocks, "d1", size=150)  # заведомо меньше трёх строк вместе
    assert len(chunks) > 1
    found = [line for c in chunks for line in c["text"].split("\n")]
    assert set(found) == expected_rows
    assert len(found) == len(expected_rows)  # ни одна строка не продублирована и не разорвана


def test_every_chunk_has_document_metadata():
    for parse, name in (
        (pdf.parse_pdf, "blender_passport.pdf"),
        (docx.parse_docx, "blender_kp.docx"),
        (xlsx.parse_xlsx, "kettle_spec.xlsx"),
    ):
        for chunk in chunk_blocks(parse(_bytes(name)), "d1"):
            assert chunk["metadata"]["page"] >= 1
            assert chunk["metadata"]["section"]
            assert chunk["metadata"]["doc_id"] == "d1"


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


def test_scanned_pdf_raises_dedicated_no_text_layer_error():
    from app.parsers.pdf import PdfNoTextLayer

    with pytest.raises(PdfNoTextLayer, match="нет текстового слоя"):
        pdf.parse_pdf(_bytes("boiler_scan.pdf"))
    assert issubclass(PdfNoTextLayer, DocumentRejected)  # сервис ловит его как отказ документа


def test_parse_to_chunks_returns_chunks_with_metadata_for_every_format():
    from app.services.documents import parse_to_chunks

    for name in ("blender_passport.pdf", "blender_kp.docx", "kettle_spec.xlsx"):
        chunks = parse_to_chunks(_bytes(name), name.rsplit(".", 1)[1], "d1")
        assert chunks
        assert all(c["metadata"]["page"] >= 1 and c["metadata"]["section"] for c in chunks)
        assert [c["ordinal"] for c in chunks] == list(range(len(chunks)))  # нумерация подряд
        assert [c["chunk_id"] for c in chunks] == [f"d1-{i}" for i in range(len(chunks))]


def test_parse_to_chunks_rejects_scan_and_unknown_format():
    from app.services.documents import parse_to_chunks

    with pytest.raises(DocumentRejected, match="нет текстового слоя"):
        parse_to_chunks(_bytes("boiler_scan.pdf"), "pdf", "d1")
    with pytest.raises(DocumentRejected, match="не поддерживается"):
        parse_to_chunks(b"text", "txt", "d1")
