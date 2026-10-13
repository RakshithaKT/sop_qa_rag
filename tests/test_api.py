from fastapi.testclient import TestClient
from sop_qa.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_endpoint():
    response = client.post(
"/api/v1/query",
json={"question": "How do I operate the machine?"},
)


    assert response.status_code == 200
    assert response.json() == {
    "answer": "Your question was received: How do I operate the machine?"
}


def test_query_rejects_empty_question():
    response = client.post(
"/api/v1/query",
json={"question": ""},
)


    assert response.status_code == 422

