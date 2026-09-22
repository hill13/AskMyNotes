import pytest

from askmynotes.extract import ExtractionError, extract_pages


def _write_pdf(path, pages):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_extract_pages_roundtrip(tmp_path):
    pdf = tmp_path / "notes.pdf"
    _write_pdf(pdf, ["Lecture one: gradients", "Lecture two: convolutions"])
    pages = extract_pages(pdf)
    assert [p.number for p in pages] == [1, 2]
    assert "gradients" in pages[0].text
    assert "convolutions" in pages[1].text


def test_empty_pdf_raises(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["", ""])
    with pytest.raises(ExtractionError, match="scanned"):
        extract_pages(pdf)


def test_corrupt_pdf_raises_extraction_error(tmp_path):
    """A broken file must surface as ExtractionError so a batch can continue."""
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(b"this is not a PDF at all, just text\n")
    with pytest.raises(ExtractionError, match="not a readable PDF"):
        extract_pages(pdf)
