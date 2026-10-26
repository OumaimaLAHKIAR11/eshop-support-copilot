import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api import routes
from backend.app.generation import generation as g
from backend.app.nlp.classification import TicketAnalysis, analyze_ticket
from backend.app.nlp import classification

MESSAGE = 'Thank you for your excellent service! My order arrived perfectly'
PRAISE = {'category': 'delivery_issue', 'sentiment': 'positive', 'priority': 'low',
          'intent': 'acknowledgement'}


@pytest.mark.parametrize('language,fragment', [('en', 'Thank you for your feedback!'), ('fr', 'Merci pour votre retour !')])
def test_praise_skips_retrieval_and_all_generation_models(monkeypatch, language, fragment):
    monkeypatch.setattr(routes, 'analyze_ticket', lambda text: PRAISE)
    search = Mock(side_effect=AssertionError('Praise must not search policies'))
    model = Mock(side_effect=AssertionError('Praise needs no generation call'))
    monkeypatch.setattr(routes, 'ticket_context', search)
    monkeypatch.setattr(g, '_get_generator', model)
    monkeypatch.setattr(g, '_get_verifier', model)
    response = TestClient(app).post('/api/v1/process-ticket', json={
        'ticket_id': 'PRAISE-1', 'message': MESSAGE, 'language': language,
        'history': [{'role': 'user', 'content': 'Where is my order?'}]})
    assert response.status_code == 200
    result = response.json()
    assert fragment in result['suggested_reply']
    assert result['needs_human_review'] is False
    assert result['sources'] == []
    search.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize('message', [
    'Thanks, but my parcel never arrived.',
    'Great service! Can I return this item?',
    'Thank you, but my refund is missing.',
])
def test_mixed_praise_is_not_a_social_reply(message):
    analysis = {**PRAISE, 'intent': 'support_request'}
    assert g.conversational_reply(message, analysis, 'en') is None


def test_urgent_message_overrides_social_intent():
    assert g.conversational_reply('Thanks, the product caused a fire.', PRAISE, 'en') is None


def test_greeting_without_policy_context():
    analysis = {**PRAISE, 'intent': 'greeting', 'sentiment': 'neutral'}
    result = g.generate_reply('Bonjour', analysis, {}, 'fr')
    assert 'Bonjour' in result['suggested_reply']
    assert not result['needs_human_review']


def test_classifier_preserves_validated_intent(monkeypatch):
    model = Mock()
    model.invoke.return_value = PRAISE
    monkeypatch.setattr(classification, '_get_classifier', lambda: model)
    assert analyze_ticket(MESSAGE)['intent'] == 'acknowledgement'


@pytest.mark.skipif(__import__('os').getenv('RUN_LLM_TESTS') != '1', reason='Opt-in live Azure intent regression')
@pytest.mark.parametrize('message,intent', [
    (MESSAGE, 'acknowledgement'),
    ('Thanks, but my parcel never arrived.', 'support_request'),
    ('Great service! Can I return this item?', 'support_request'),
])
def test_live_conversational_intent(message, intent):
    assert analyze_ticket(message)['intent'] == intent
