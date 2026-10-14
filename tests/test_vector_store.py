
from sop_qa.VectorDb import vector_store as vector_store_module
from sop_qa.VectorDb.vector_store import VectorStore


def fake_embeddings(texts):
    vectors = {
        "Inspect guards before startup.": [1.0, 0.0, 0.0],
        "Wear protective gloves.": [0.0, 1.0, 0.0],
    }
    return [vectors[text] for text in texts]


def test_add_and_search_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vector_store_module,
        "embed_texts",
        fake_embeddings,
    )

    store = VectorStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_chunks",
    )

    try:
        chunks = [
            {
                "chunk_id": "doc1-chunk-1",
                "filename": "machine.pdf",
                "page_number": 1,
                "text": "Inspect guards before startup.",
            },
            {
                "chunk_id": "doc1-chunk-2",
                "filename": "safety.pdf",
                "page_number": 2,
                "text": "Wear protective gloves.",
            },
        ]

        assert store.add_chunks(chunks) == 2

        results = store.search(
            "Inspect guards before startup.",
            limit=2,
        )

        assert len(results) == 2
        assert results[0]["chunk_id"] == "doc1-chunk-1"
        assert results[0]["filename"] == "machine.pdf"
        assert results[0]["page_number"] == 1
        assert results[0]["score"] > 0.99

    finally:
        store.close()


def test_empty_search_question_is_rejected(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "qdrant"),
        collection_name="empty_query_test",
    )

    try:
        import pytest

        with pytest.raises(ValueError, match="empty"):
            store.search("   ")

    finally:
        store.close()
