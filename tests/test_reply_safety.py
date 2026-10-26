from unittest.mock import Mock
import pytest
from backend.app.generation import generation as g
from tests.test_generation import install, ANALYSIS, CONTEXT


@pytest.mark.parametrize('ticket', ['How can I return my order?', 'Comment retourner ma commande ?'])
def test_general_question_not_escalated_by_possessive(monkeypatch, ticket):
    install(monkeypatch)
    assert not g.generate_reply(ticket, ANALYSIS, CONTEXT)['needs_human_review']


def test_review_draft_never_reaches_customer(monkeypatch):
    install(monkeypatch, suggested_reply='Your refund is approved [return_policy.txt:1].', requires_order_verification=True)
    result = g.generate_reply('Refund policy?', ANALYSIS, CONTEXT)
    assert result['needs_human_review']
    assert 'approved' not in result['suggested_reply']
    assert not result['sources']


def test_semantically_invalid_citation_blocks_draft(monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(g, 'verify_reply', lambda *args: False)
    assert g.generate_reply('Policy?', ANALYSIS, CONTEXT)['needs_human_review']


@pytest.mark.parametrize('field', ['all_claims_supported', 'citations_support_claims', 'all_questions_answered', 'contains_unverified_action_or_status'])
def test_verifier_rejects_each_failure(monkeypatch, field):
    payload = dict(all_claims_supported=True, citations_support_claims=True,
                   all_questions_answered=True, contains_unverified_action_or_status=False)
    payload[field] = not payload[field]
    model = Mock()
    model.invoke.return_value = payload
    monkeypatch.setattr(g, '_get_verifier', lambda: model)
    assert not g.verify_reply('Policy?', 'Claim [return_policy.txt:1]', CONTEXT['results'])
