from unittest.mock import Mock
from backend.app.generation import generation as g
from backend.app.rag.ingest import build_chunks
from frontend.reply_format import customer_reply_text, source_label

RETURN = 'I would like to know how to return an item.'
DELAY = 'This is unacceptable! My package is two weeks late and nobody is helping me'


def test_general_return_is_concise_and_never_needs_model_review(monkeypatch):
    model = Mock(side_effect=AssertionError('Routine procedure needs no model call'))
    monkeypatch.setattr(g, '_get_generator', model)
    context = {'is_relevant': True, 'results': [dict(text=doc.page_content, **doc.metadata) for doc in build_chunks()]}
    result = g.generate_reply(RETURN, {'category': 'return_request', 'priority': 'medium'}, context)
    assert not result['needs_human_review']
    assert len(result['suggested_reply'].split()) < 120
    assert '30 calendar days' in result['suggested_reply']
    assert 'please provide your order number' in result['suggested_reply']
    assert 'refund' not in result['suggested_reply'].lower()
    assert len(result['sources']) == 4
    model.assert_not_called()


def test_assistant_request_for_number_is_not_customer_supplied_number():
    history = [{'role': 'user', 'content': RETURN}, {'role': 'assistant', 'content':
        'Please reply with your order number so support can verify your order.'}]
    assert g.customer_order_number(DELAY, history) is None
    result = g.support_handoff(DELAY, {'category': 'delivery_issue'}, 'en', history)
    assert 'your order number' in result['suggested_reply']


def test_bare_words_are_not_order_identifiers():
    for text in ['order number so support can help', 'order number please', 'order number is missing']:
        assert g.customer_order_number(text) is None


def test_customer_identifier_is_recognized():
    history = [{'role': 'user', 'content': 'My order number is AB-12345'}]
    assert g.customer_order_number(DELAY, history) == 'AB-12345'
    assert g.customer_order_number('My order is NS-12345') == 'NS-12345'


def test_assistant_invented_number_is_not_customer_evidence():
    assert g.customer_order_number(DELAY, [{'role': 'assistant', 'content': 'Your order is NS-12345'}]) is None


def test_customer_prose_has_no_internal_document_ids():
    sources = [{'document': 'return_policy.txt', 'section': '1'}, {'document': 'return_policy.txt', 'section': '2'}]
    result = customer_reply_text('Return within 30 days [return_policy.txt:1][return_policy.txt:2]. Keep [this] text.', sources)
    assert result == 'Return within 30 days. Keep [this] text.'
    assert source_label(sources[0]) == 'Return policy — rule 1'
