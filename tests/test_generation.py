from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from backend.app.generation import generation as g

CONTEXT = {"is_relevant": True, "results": [
    {"text": "Returns are accepted within 30 days.", "document": "return_policy.txt", "section": "1"},
    {"text": "Unused items only.", "document": "return_policy.txt", "section": "2"},
]}
ANALYSIS = {"category": "return_request", "sentiment": "neutral", "priority": "low"}


def install(monkeypatch, **overrides):
    monkeypatch.setattr(g, "verify_reply", lambda *args: True)
    payload = dict(suggested_reply="Returns are accepted within 30 days [return_policy.txt:1].",
                   used_source_ids=["return_policy.txt:1"], context_sufficient=True,
                   requires_order_verification=False, requires_human_review=False)
    payload.update(overrides)
    calls = []
    monkeypatch.setattr(g, "_get_generator", lambda: RunnableLambda(lambda value: calls.append(value) or payload))
    return calls


def test_general_policy_uses_only_cited_sources(monkeypatch):
    install(monkeypatch)
    result = g.generate_reply("What is the return window?", ANALYSIS, CONTEXT)
    assert result["needs_human_review"] is False
    assert result["sources"] == [{"document": "return_policy.txt", "section": "1"}]


@pytest.mark.parametrize("overrides", [
    {"context_sufficient": False}, {"requires_order_verification": True},
    {"requires_human_review": True},
])
def test_model_review_signals_are_preserved(monkeypatch, overrides):
    install(monkeypatch, **overrides)
    assert g.generate_reply("Return policy?", ANALYSIS, CONTEXT)["needs_human_review"]


@pytest.mark.parametrize("ticket", ["Where is my order?", "Quel est le statut de ma commande ?", "The product caused a fire."])
def test_operational_and_safety_cases_cannot_be_auto_approved(monkeypatch, ticket):
    install(monkeypatch)
    assert g.generate_reply(ticket, ANALYSIS, CONTEXT)["needs_human_review"]


@pytest.mark.parametrize("overrides", [
    {"suggested_reply": "A claim without citation."},
    {"suggested_reply": "Claim [unknown:1].", "used_source_ids": ["unknown:1"]},
])
def test_invalid_citations_fall_back(monkeypatch, overrides):
    install(monkeypatch, **overrides)
    result = g.generate_reply("Return policy?", ANALYSIS, CONTEXT)
    assert result["needs_human_review"] and result["sources"] == []


def test_no_context_skips_model_and_uses_french(monkeypatch):
    model = Mock(side_effect=AssertionError("must not call model"))
    monkeypatch.setattr(g, "_get_generator", model)
    result = g.generate_reply("Bonjour", ANALYSIS, {}, language="fr")
    assert "Merci" in result["suggested_reply"]
    model.assert_not_called()


def test_french_prompt(monkeypatch):
    calls = install(monkeypatch)
    g.generate_reply("Quelle est la politique de retour ?", ANALYSIS, CONTEXT, language="fr")
    assert "French" in calls[0].to_string()


def test_model_failure_falls_back(monkeypatch):
    monkeypatch.setattr(g, "_get_generator", Mock(side_effect=RuntimeError("offline")))
    assert g.generate_reply("Policy?", ANALYSIS, CONTEXT)["needs_human_review"]

def test_document_only_source_ids_resolve_from_valid_inline_citations(monkeypatch):
    install(monkeypatch, used_source_ids=['return_policy.txt'])
    result = g.generate_reply('Return policy?', ANALYSIS, CONTEXT)
    assert not result['needs_human_review']
    assert result['sources'] == [{'document': 'return_policy.txt', 'section': '1'}]


def test_document_name_cannot_authorize_unknown_section(monkeypatch):
    install(monkeypatch, used_source_ids=['return_policy.txt'],
            suggested_reply='Unsupported claim [return_policy.txt:99].')
    result = g.generate_reply('Return policy?', ANALYSIS, CONTEXT)
    assert result['needs_human_review']
    assert not result['sources']

def test_known_unused_metadata_does_not_discard_valid_cited_answer(monkeypatch):
    install(monkeypatch, used_source_ids=['return_policy.txt:1', 'return_policy.txt:2'])
    result = g.generate_reply('Return policy?', ANALYSIS, CONTEXT)
    assert not result['needs_human_review']
    assert result['sources'] == [{'document': 'return_policy.txt', 'section': '1'}]

@pytest.mark.parametrize('separator', ['; ', ', '])
def test_grouped_valid_citations_are_accepted_and_normalized(monkeypatch, separator):
    install(monkeypatch, used_source_ids=['return_policy.txt'],
            suggested_reply='Return conditions [' + separator.join(['return_policy.txt:1', 'return_policy.txt:2']) + '].')
    result = g.generate_reply('Return policy?', ANALYSIS, CONTEXT)
    assert not result['needs_human_review']
    assert len(result['sources']) == 2
    assert '[return_policy.txt:1][return_policy.txt:2]' in result['suggested_reply']


@pytest.mark.parametrize('metadata', [[], ['unknown:1'], ['return_policy.txt:2']])
def test_inline_evidence_is_authoritative_over_redundant_metadata(monkeypatch, metadata):
    install(monkeypatch, used_source_ids=metadata)
    result = g.generate_reply('Return policy?', ANALYSIS, CONTEXT)
    assert not result['needs_human_review']
    assert result['sources'] == [{'document': 'return_policy.txt', 'section': '1'}]
