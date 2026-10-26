from unittest.mock import Mock
from backend.app.rag import retriever


def test_neighbor_expansion_includes_procedure_conditions(monkeypatch):
    hit = {'text': 'Shipping instructions follow review.', 'document': 'return_policy.txt',
           'section': '6', 'score': 0.6}
    store = Mock()
    store.get.return_value = {
        'documents': ['Return within 30 days.', 'Provide order number.', 'Shipping instructions follow review.'],
        'metadatas': [{'document': 'return_policy.txt', 'section': section} for section in ['1', '3', '6']]}
    monkeypatch.setattr(retriever, '_get_store', lambda *args: store)
    monkeypatch.setattr(retriever, 'active_collection', lambda *args: 'test')
    results = retriever.expand_policy_context([hit])
    assert {(r['document'], r['section']) for r in results} == {
        ('return_policy.txt', '1'), ('return_policy.txt', '3'), ('return_policy.txt', '6')}
    assert results[0] == hit
    assert results[1]['retrieval_method'] == 'policy_neighbor'


def test_no_hits_does_not_fetch_unrelated_policies(monkeypatch):
    store = Mock(side_effect=AssertionError('Must not fetch unrelated policies'))
    monkeypatch.setattr(retriever, '_get_store', store)
    assert retriever.expand_policy_context([]) == []
    store.assert_not_called()
