from unittest.mock import Mock
import pytest
from backend.app.rag import query


def test_english_needs_no_translation(monkeypatch):
    factory = Mock(side_effect=AssertionError("No model needed"))
    monkeypatch.setattr(query, "_get_translator", factory)
    assert query.prepare_search_query("Return policy?", "en") == "Return policy?"
    factory.assert_not_called()


def test_french_translates_without_answering(monkeypatch):
    model = Mock()
    model.invoke.return_value = {"english_query": "What is the return window?"}
    monkeypatch.setattr(query, "_get_translator", lambda: model)
    assert query.prepare_search_query("Quel est le delai de retour ?", "fr") == "What is the return window?"
    assert model.invoke.call_args.args[0][1][1] == "Quel est le delai de retour ?"


def test_translation_failure_propagates(monkeypatch):
    monkeypatch.setattr(query, "_get_translator", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError):
        query.prepare_search_query("Bonjour", "fr")
