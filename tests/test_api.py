
from unittest.mock import Mock

from fastapi.testclient import TestClient
from sop_qa import api
from sop_qa.main import app

client = TestClient(app)


class FakeAnswerGenerator:
    async def generate(self, question, evidence):
        return Mock(
            answer="The provided evidence does not contain relevant instructions.",
            sources=[],
        )


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_endpoint(monkeypatch):
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
        json={"question": "How do I operate the machine?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == (
        "The provided evidence does not contain relevant instructions."
    )
    assert body["sources"] == []


def test_query_rejects_empty_question():
    response = client.post(
        "/api/v1/query",
        json={"question": ""},
    )

    assert response.status_code == 422
