import pytest
from unittest.mock import Mock
from langchain_core.runnables import RunnableLambda
from backend.app.generation import generation as g
from backend.app.rag.ingest import build_chunks

DAMAGE = 'My laptop arrived with a cracked screen'
ANALYSIS = {'category': 'damaged_product', 'sentiment': 'negative', 'priority': 'medium', 'intent': 'support_request'}


def context():
    return {'is_relevant': True, 'results': [dict(text=doc.page_content, **doc.metadata) for doc in build_chunks()]}


@pytest.mark.parametrize('failure', ['invalid_source', 'verifier_rejection', 'model_exception', 'long_answer', 'no_evidence'])
def test_damaged_item_always_gets_actionable_help(monkeypatch, failure):
    payload = dict(suggested_reply='Please share your order number [damaged_products_policy.txt:1].',
                   context_sufficient=True, requires_order_verification=True, requires_human_review=True)
    if failure == 'invalid_source':
        payload['suggested_reply'] = 'Unsupported claim [unknown:99].'
    elif failure == 'long_answer':
        payload['suggested_reply'] = 'word ' * 150 + '[damaged_products_policy.txt:1]'
    monkeypatch.setattr(g, '_get_generator', lambda: RunnableLambda(lambda value: payload))
    monkeypatch.setattr(g, 'verify_reply', lambda *args: False)
    if failure == 'model_exception':
        monkeypatch.setattr(g, '_get_generator', Mock(side_effect=RuntimeError('offline')))
    result = g.generate_reply(DAMAGE, ANALYSIS, {} if failure == 'no_evidence' else context())
    text = result['suggested_reply'].lower()
    assert 'order number' in text
    assert 'photo' in text
    assert 'review the case' in text
    assert 'before providing a reliable answer' not in text
    assert len(text.split()) < 85
    assert result['needs_human_review']
    assert 'we have approved' not in text
    assert 'will replace' not in text
    if failure != 'no_evidence':
        assert result['sources']


def test_damaged_item_does_not_repeat_supplied_details():
    history = [{'role': 'user', 'content': 'My order number is NS-1234. I attached photos of the screen.'}]
    result = g.support_handoff(DAMAGE, ANALYSIS, 'en', history, context())
    assert 'Could you' not in result['suggested_reply']
    assert 'review the case' in result['suggested_reply']


def test_french_damage_handoff_is_actionable():
    result = g.support_handoff('Mon ordinateur est casse', ANALYSIS, 'fr', context=context())
    assert 'photo' in result['suggested_reply']
    assert 'commande' in result['suggested_reply']
    assert 'reliable answer' not in result['suggested_reply']


@pytest.mark.parametrize('category,expected', [('refund_request', 'approved'), ('return_request', 'Which item')])
def test_other_supported_categories_have_specific_handoffs(category, expected):
    result = g.support_handoff('Help me', {'category': category}, 'en')
    assert expected in result['suggested_reply']
    assert 'order number' in result['suggested_reply']
    assert 'before providing a reliable answer' not in result['suggested_reply']

def test_empty_retrieval_uses_category_as_query_hint(monkeypatch):
    from backend.app.api import routes
    from backend.app.schemas import TicketRequest
    queries = []
    def retrieve(query, k):
        queries.append(query)
        if 'damaged defective' in query:
            return {'results': [{'document': 'damaged_products_policy.txt', 'section': '1', 'text': 'Contact support with order number.', 'score': 0.6}]}
        return {'results': []}
    monkeypatch.setattr(routes, 'prepare_search_query', lambda text, language: text)
    monkeypatch.setattr(routes, 'retrieve_context', retrieve)
    monkeypatch.setattr(routes, 'expand_policy_context', lambda results: results)
    result = routes.ticket_context(TicketRequest(ticket_id='D1', message=DAMAGE), ANALYSIS)
    assert result['is_relevant']
    assert result['results'][0]['document'] == 'damaged_products_policy.txt'
    assert DAMAGE in queries[-1]
