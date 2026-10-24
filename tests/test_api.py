
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_ticket():
    response = client.post(
        "/api/v1/analyze",
        json={
            "ticket_id": "T001",
            "message": "My order has not arrived.",
            "language": "en",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "delivery_issue"
    assert data["sentiment"] == "negative"
    assert data["priority"] == "medium"


def test_retrieve_context():
    response = client.post(
        "/api/v1/retrieve",
        json={
            "ticket_id": "T001",
            "message": "My order has not arrived.",
            "language": "en",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["is_relevant"] is True
    assert len(data["results"]) > 0


def test_process_ticket():
    response = client.post(
        "/api/v1/process-ticket",
        json={
            "ticket_id": "T001",
            "message": "My order has not arrived.",
            "language": "en",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["ticket_id"] == "T001"
    assert data["suggested_reply"]
    assert isinstance(data["sources"], list)
    assert isinstance(data["needs_human_review"], bool)


def test_empty_ticket_message():
    response = client.post(
        "/api/v1/process-ticket",
        json={
            "ticket_id": "T001",
            "message": "   ",
            "language": "en",
        },
    )

    assert response.status_code == 400
