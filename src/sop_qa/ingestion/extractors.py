
from io import BytesIO

from pypdf import PdfReader


def extract_pdf(file_bytes: bytes, filename: str) -> dict:
    """Extract text from a PDF while preserving page numbers."""

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    return {
        "filename": filename,
        "page_count": len(pages),
        "pages": pages,
    }