import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from backend.app.main import app
from backend.app.api import routes
from backend.app.generation import generation as g
from backend.app.schemas import TicketRequest

PRAISE = 'Thank you for your excellent service! My order arrived perfectly'
ANGRY = 'This is unacceptable! My package is two weeks late and nobody is helping me'
RETURN = 'I would like to know how to return an item.'


def test_self_contained_return_ignores_old_delivery_complaint():
    request = TicketRequest(ticket_id='T1', message=RETURN, history=[
        {'role': 'user', 'content': PRAISE}, {'role': 'user', 'content': ANGRY}])
    assert routes.customer_context(request) == RETURN


def test_followup_still_uses_previous_topic():
    request = TicketRequest(ticket_id='T1', message='And how long does it take?', history=[
        {'role': 'user', 'content': RETURN}])
    assert RETURN in routes.customer_context(request)


def test_high_priority_can_give_verified_guidance(monkeypatch):
    draft = "I'm sorry for the frustrating delay. Please share your order number [delivery_policy.txt:9]. Support needs to check the shipment."
    payload = {'suggested_reply': draft, 'used_source_ids': ['delivery_policy.txt:9'],
               'context_sufficient': True, 'requires_order_verification': True, 'requires_human_review': True}
    monkeypatch.setattr(g, '_get_generator', lambda: RunnableLambda(lambda value: payload))
    monkeypatch.setattr(g, 'verify_reply', lambda *args: True)
    result = g.generate_reply(ANGRY, {'category': 'delivery_issue', 'priority': 'high'}, {
        'is_relevant': True, 'results': [{'document': 'delivery_policy.txt', 'section': '9',
                                        'text': 'Ask for the order number before investigating.'}]})
    assert result['suggested_reply'] == draft
    assert result['needs_human_review']
    assert result['sources']


def test_delivery_failure_still_provides_a_concrete_next_step():
    result = g.generate_reply(ANGRY, {'category': 'delivery_issue', 'priority': 'high'}, {})
    assert 'order number' in result['suggested_reply']
    assert 'latest tracking' in result['suggested_reply']
    assert result['needs_human_review']


def test_reported_conversation_does_not_contaminate_return(monkeypatch):
    monkeypatch.setattr(routes, "expand_policy_context", lambda results: results)
    seen = []
    def classify(message):
        seen.append(message)
        return {'category': 'return_request', 'sentiment': 'neutral', 'priority': 'low', 'intent': 'support_request'}
    def search(message, k):
        assert ANGRY not in message
        return {'is_relevant': True, 'results': [{'text': 'Provide the order number to start a return.',
                                                'document': 'return_policy.txt', 'section': '3', 'score': 0.8}]}
    monkeypatch.setattr(routes, 'analyze_ticket', classify)
    monkeypatch.setattr(routes, 'retrieve_context', search)
    monkeypatch.setattr(g, '_get_generator', lambda: RunnableLambda(lambda value: {
        'suggested_reply': 'To start a return, provide your order number [return_policy.txt:3].',
        'used_source_ids': ['return_policy.txt:3'], 'context_sufficient': True,
        'requires_order_verification': False, 'requires_human_review': False}))
    monkeypatch.setattr(g, 'verify_reply', lambda *args: True)
    response = TestClient(app).post('/api/v1/process-ticket', json={
        'ticket_id': 'RETURN-AFTER-ANGER', 'message': RETURN,
        'history': [{'role': 'user', 'content': PRAISE}, {'role': 'user', 'content': ANGRY}]})
    assert response.status_code == 200
    assert seen == [RETURN]
    assert 'order number' in response.json()['suggested_reply']
    assert not response.json()['needs_human_review']

def test_rejected_draft_recovers_procedure_from_exact_evidence(monkeypatch):
    from backend.app.rag.ingest import build_chunks
    chunks = [dict(text=doc.page_content, **doc.metadata) for doc in build_chunks()]
    monkeypatch.setattr(g, '_get_generator', lambda: RunnableLambda(lambda value: {
        'suggested_reply': 'Unsupported text [return_policy.txt:99].',
        'used_source_ids': [], 'context_sufficient': True,
        'requires_order_verification': False, 'requires_human_review': False}))
    result = g.generate_reply(RETURN, {'category': 'return_request', 'priority': 'medium'},
                              {'is_relevant': True, 'results': chunks})
    assert '30 calendar days' in result['suggested_reply']
    assert 'order number' in result['suggested_reply']
    assert not result['needs_human_review']
    assert len(result['sources']) == 4


def test_extractive_recovery_does_not_answer_compound_question_partially():
    assert g.extractive_procedure_reply('How do I return an item and get a refund?',
        {'category': 'return_request'}, {}, 'en') is None
