import os
import pytest
from backend.app.nlp.classification import TicketAnalysis, analyze_ticket


def test_schema_accepts_only_allowed_labels():
    result = TicketAnalysis(
        category="delivery_issue", sentiment="negative", priority="high"
    )
    assert result.category == "delivery_issue"


def test_schema_rejects_unknown_category():
    with pytest.raises(Exception):
        TicketAnalysis(category="shipping", sentiment="negative", priority="high")


@pytest.mark.parametrize("bad_text", ["", "   ", None, 123])
def test_empty_or_invalid_ticket_raises_value_error(bad_text):
    with pytest.raises(ValueError):
        analyze_ticket(bad_text)


@pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="Set RUN_LLM_TESTS=1 to call the paid OpenAI API for integration tests.",
)
@pytest.mark.parametrize("text", [
    "My parcel is late and tracking has not changed for four days.",
    "I want a refund for my order.",
    "How can I return an unused item?",
    "The headphones arrived broken.",
    "Thank you, the service was excellent!",
])
def test_live_classification_returns_valid_labels(text):
    result = analyze_ticket(text)
    assert set(result) == {"category", "sentiment", "priority"}
    TicketAnalysis.model_validate(result)
