import pytest
from backend.app.conversation import resolve_case
from backend.app.schemas import TicketRequest
from backend.app.api import routes
from backend.app.generation import generation as g
from fastapi.testclient import TestClient

ISSUE = 'My laptop arrived with a cracked screen.'
NUMBER = 'My order number is NS-12345'
QUESTION = 'What else do you need from me?'


def history(*messages):
    return [{'role': 'user', 'content': message} for message in messages]


@pytest.mark.parametrize('message', [NUMBER, 'NS-12345', QUESTION, 'How can I return it?'])
def test_followups_preserve_active_issue(message):
    case = resolve_case(message, history(ISSUE))
    assert case.is_followup
    assert case.issue == ISSUE
    assert ISSUE in case.text
    assert message in case.text


def test_details_only_turns_do_not_replace_original_issue():
    case = resolve_case(QUESTION, history(ISSUE, NUMBER))
    assert case.issue == ISSUE
    assert NUMBER in case.text


@pytest.mark.parametrize('message', ['I would like to know how to return an item.', 'My package is two weeks late.', 'My phone arrived broken.'])
def test_new_explicit_issue_does_not_inherit_old_issue(message):
    assert resolve_case(message, history(ISSUE, NUMBER)).text == message


def test_assistant_claims_are_not_case_facts():
    case = resolve_case(QUESTION, history(ISSUE) + [{'role': 'assistant', 'content': 'Refund approved, NS-99999'}])
    assert 'approved' not in case.text
    assert 'NS-99999' not in case.text


def test_followup_without_an_issue_does_not_invent_one():
    assert not resolve_case(NUMBER, []).is_followup


def test_damaged_followup_does_not_repeat_number_or_apology():
    analysis = {'category': 'damaged_product', 'case_followup': True}
    result = g.support_handoff(QUESTION, analysis, 'en', history(ISSUE, NUMBER))
    assert "already provided your order number" in result['suggested_reply']
    assert 'photo' in result['suggested_reply']
    assert 'Could you share your order number' not in result['suggested_reply']
    assert "I'm sorry" not in result['suggested_reply']


def test_exact_three_turn_sequence_keeps_category_and_requests_only_missing_details(monkeypatch):
    def classify(text):
        assert ISSUE in text
        return {'category': 'damaged_product', 'sentiment': 'neutral', 'priority': 'medium', 'intent': 'support_request'}
    monkeypatch.setattr(routes, 'analyze_ticket', classify)
    monkeypatch.setattr(routes, 'prepare_search_query', lambda text, language: text)
    monkeypatch.setattr(routes, 'retrieve_context', lambda text, k: {'results': [
        {'text': 'Request photos as evidence.', 'document': 'damaged_products_policy.txt', 'section': '2', 'score': 0.8}]})
    monkeypatch.setattr(routes, 'expand_policy_context', lambda results: results)
    for message, previous in [(NUMBER, history(ISSUE)), (QUESTION, history(ISSUE, NUMBER))]:
        response = TestClient(__import__('backend.app.main', fromlist=['app']).app).post('/api/v1/process-ticket', json={
            'ticket_id': 'CASE-1', 'message': message, 'history': previous})
        assert response.status_code == 200
        result = response.json()
        assert result['category'] == 'damaged_product'
        assert 'photo' in result['suggested_reply']
        assert 'Could you share your order number' not in result['suggested_reply']
        assert 'reliable answer' not in result['suggested_reply']
        assert result['sources']

def test_ready_photos_are_a_case_update():
    case = resolve_case('I have photos ready.', history(ISSUE, NUMBER))
    assert case.is_followup
    assert case.issue == ISSUE


def test_customer_correction_overrides_earlier_photo_claim():
    assert not g.customer_photos_ready("I don't have photos yet.", history('I have photos ready.'))
    assert g.customer_photos_ready('I have photos ready.', history(ISSUE, NUMBER))


def test_complete_details_are_not_requested_again():
    previous = history(ISSUE, NUMBER, 'I have photos ready.')
    result = g.support_handoff(QUESTION, {'category': 'damaged_product', 'case_followup': True}, 'en', previous)
    assert 'Could you' not in result['suggested_reply']
    assert 'keep a clear photo' not in result['suggested_reply']
    assert 'review' in result['suggested_reply']

def test_conversation_progresses_from_number_to_missing_evidence_to_review():
    analysis = {'category': 'damaged_product', 'case_followup': True}
    first = g.support_handoff(NUMBER, analysis, 'en', history(ISSUE))
    second = g.support_handoff(QUESTION, analysis, 'en', history(ISSUE, NUMBER))
    third = g.support_handoff('I have photos ready.', analysis, 'en', history(ISSUE, NUMBER, QUESTION))
    fourth = g.support_handoff(QUESTION, analysis, 'en', history(ISSUE, NUMBER, QUESTION, 'I have photos ready.'))
    assert 'NS-12345' in first['suggested_reply']
    assert 'next thing to prepare' in second['suggested_reply']
    assert first['suggested_reply'] != second['suggested_reply']
    assert 'Thanks' in third['suggested_reply']
    assert 'review' in fourth['suggested_reply']
    assert 'next thing to prepare' not in fourth['suggested_reply']
    assert 'Could you share' not in fourth['suggested_reply']
    assert 'have been approved' not in fourth['suggested_reply']


def test_corrected_order_number_is_acknowledged_without_reusing_old_number():
    result = g.support_handoff('My order number is NS-99999',
        {'category': 'damaged_product', 'case_followup': True}, 'en', history(ISSUE, NUMBER))
    assert 'NS-99999' in result['suggested_reply']
    assert 'NS-12345' not in result['suggested_reply']


def test_action_question_is_not_replaced_by_a_missing_details_template():
    assert g.damaged_case_step('How can I return it?',
        {'category': 'damaged_product', 'case_followup': True}, 'en', history(ISSUE, NUMBER), {}) is None
