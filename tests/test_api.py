import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api import routes

client = TestClient(app)


@pytest.fixture(autouse=True)
def services(monkeypatch):
    monkeypatch.setattr(routes, "expand_policy_context", lambda results: results)
    monkeypatch.setattr(routes, "prepare_search_query", lambda text, language: "Translated query" if language == "fr" else text)
    monkeypatch.setattr(routes, "analyze_ticket", lambda text: {
        "category": "delivery_issue", "sentiment": "negative", "priority": "medium"})
    monkeypatch.setattr(routes, "retrieve_context", lambda text, k: {
        "is_relevant": True, "results": [{"text": "Policy", "document": "delivery_policy.txt", "section": "1", "score": 0.8}]})


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_ticket():
    response = client.post("/api/v1/analyze", json={"ticket_id": "T1", "message": "Late parcel"})
    assert response.status_code == 200
    assert response.json()["category"] == "delivery_issue"


def test_retrieve_context():
    response = client.post("/api/v1/retrieve", json={"ticket_id": "T1", "message": "Late parcel"})
    assert response.status_code == 200
    assert response.json()["is_relevant"]


def test_process_ticket_forwards_language(monkeypatch):
    def generate(**kwargs):
        assert kwargs["language"] == "fr"
        assert kwargs["context"]["is_relevant"]
        return {"suggested_reply": "Bonjour", "sources": [], "needs_human_review": True}
    monkeypatch.setattr(routes, "generate_reply", generate)
    response = client.post("/api/v1/process-ticket", json={"ticket_id": "T1", "message": "Mon colis", "language": "fr"})
    assert response.status_code == 200
    assert response.json()["needs_human_review"]


@pytest.mark.parametrize("endpoint", ["analyze", "retrieve", "process-ticket"])
def test_empty_ticket_message(endpoint):
    assert client.post(f"/api/v1/{endpoint}", json={"ticket_id": "T1", "message": " "}).status_code == 400


def test_unsupported_language():
    assert client.post("/api/v1/process-ticket", json={"ticket_id": "T1", "message": "Hi", "language": "xx"}).status_code == 422


def test_service_failure(monkeypatch):
    def fail(text, k):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(routes, "retrieve_context", fail)
    assert client.post("/api/v1/process-ticket", json={"ticket_id": "T1", "message": "Hi"}).status_code == 503


def test_no_duplicate_routes():
    keys = [(route.path, method) for route in app.routes for method in getattr(route, "methods", [])]
    assert len(keys) == len(set(keys))

def test_translation_failure_returns_safe_reply(monkeypatch):
    def fail(*args):
        raise RuntimeError('translator offline')
    monkeypatch.setattr(routes, 'prepare_search_query', fail)
    response = client.post('/api/v1/process-ticket', json={
        'ticket_id': 'T2', 'message': 'Bonjour', 'language': 'fr'})
    assert response.status_code == 200
    assert response.json()['needs_human_review']
    assert response.json()['sources'] == []


def test_history_reaches_search_and_generation(monkeypatch):
    searches = []
    def prepare(text, language):
        searches.append(text)
        return text
    def generate(**kwargs):
        assert kwargs['history'][0]['content'] == 'How do returns work?'
        assert kwargs['ticket'] == 'And how long does it take?'
        return {'suggested_reply': 'Reply', 'sources': [], 'needs_human_review': False}
    monkeypatch.setattr(routes, 'prepare_search_query', prepare)
    monkeypatch.setattr(routes, 'generate_reply', generate)
    response = client.post('/api/v1/process-ticket', json={
        'ticket_id': 'T3', 'message': 'And how long does it take?',
        'history': [{'role': 'user', 'content': 'How do returns work?'},
                    {'role': 'assistant', 'content': 'Your refund was approved.'}]})
    assert response.status_code == 200
    assert 'How do returns work?' in searches[0]
    assert 'Your refund was approved.' not in searches[0]


def test_history_is_bounded():
    response = client.post('/api/v1/process-ticket', json={
        'ticket_id': 'T4', 'message': 'Hi',
        'history': [{'role': 'user', 'content': 'Hi'}] * 11})
    assert response.status_code == 422
