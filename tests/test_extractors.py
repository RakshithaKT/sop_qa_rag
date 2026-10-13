
from io import BytesIO

import pytest
from pypdf import PdfWriter

from sop_qa.ingestion.extractors import extract_pdf


def make_test_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    output = BytesIO()
    writer.write(output)

    return output.getvalue()


def test_extract_pdf_preserves_page_count():
    result = extract_pdf(make_test_pdf(), "test.pdf")

    assert result["filename"] == "test.pdf"
    assert result["page_count"] == 1
    assert result["pages"][0]["page_number"] == 1


def test_extract_pdf_rejects_non_pdf_filename():
    with pytest.raises(ValueError, match="Only PDF"):
        extract_pdf(b"not a PDF", "document.txt")


def test_extract_pdf_rejects_invalid_pdf():
    with pytest.raises(Exception):
        extract_pdf(b"not a valid PDF", "document.pdf")