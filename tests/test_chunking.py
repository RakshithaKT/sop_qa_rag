
import pytest

from sop_qa.ingestion.chunking import chunk_document


def test_empty_document_returns_no_chunks():
    document = {
        "filename": "test.pdf",
        "pages": [{"page_number": 1, "text": ""}],
    }

    assert chunk_document(document) == []


def test_short_page_produces_one_chunk():
    document = {
        "filename": "test.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Inspect the machine before startup.",
            }
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["filename"] == "test.pdf"
    assert chunks[0]["text"] == (
        "Inspect the machine before startup."
    )


def test_long_page_produces_overlapping_chunks():
    text = " ".join(
        f"Procedure step {i} requires inspection."
        for i in range(100)
    )

    document = {
        "filename": "machine.pdf",
        "pages": [{"page_number": 2, "text": text}],
    }

    chunks = chunk_document(
        document,
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 300 for chunk in chunks)
    assert all(chunk["page_number"] == 2 for chunk in chunks)
    assert all(chunk["filename"] == "machine.pdf" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_chunk_configuration(chunk_size, overlap):
    document = {
        "filename": "test.pdf",
        "pages": [{"page_number": 1, "text": "Some text."}],
    }

    with pytest.raises(ValueError):
        chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
