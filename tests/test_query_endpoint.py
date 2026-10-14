
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from sop_qa import api
from sop_qa.generation.response import (
    GeneratedAnswer,
    RetrievedEvidence,
)
from sop_qa.main import app


client = TestClient(app)


class FakeAnswerGenerator:
    async def generate(self, question, evidence):
        if not evidence:
            return GeneratedAnswer(
                answer=(
                    "I could not find relevant evidence in the "
                    "indexed SOP documents to answer this question."
                ),
                sources=[],
            )

        return GeneratedAnswer(
            answer="Complete the startup checklist first. [S1]",
            sources=evidence,
        )


@pytest.fixture
def mock_rag(monkeypatch):
    monkeypatch.setattr(
        api.vector_store,
        "search",
        Mock(
            return_value=[
                {
                    "text": "Complete the startup checklist before operation.",
                    "source": "machine_sop.pdf",
                    "page": 3,
                    "chunk_id": "chunk-001",
                    "score": 0.87,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        api,
        "get_answer_generator",
        lambda: FakeAnswerGenerator(),
    )


def test_query_returns_generated_answer_and_sources(mock_rag):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What should I do before operation?"
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == (
        "Complete the startup checklist first. [S1]"
    )
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "machine_sop.pdf"
    assert data["sources"][0]["page"] == 3
    assert data["sources"][0]["chunk_id"] == "chunk-001"
    assert data["sources"][0]["score"] == pytest.approx(0.87)


def test_query_returns_no_sources_when_nothing_is_retrieved(
    monkeypatch,
):
    monkeypatch.setattr(
        api.vector_store,
        "search",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        api,
        "get_answer_generator",
        lambda: FakeAnswerGenerator(),
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is the startup procedure?"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "could not find relevant evidence" in (
        response.json()["answer"]
    )


def test_query_rejects_empty_question():
    response = client.post(
        "/api/v1/query",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_query_rejects_question_over_limit():
    response = client.post(
        "/api/v1/query",
        json={"question": "x" * 2001},
    )

    assert response.status_code == 422
