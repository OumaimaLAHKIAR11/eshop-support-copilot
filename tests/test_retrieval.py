
from backend.app.rag.retriever import retrieve_context


def test_delivery_question_returns_relevant_context():
    result = retrieve_context("My package has not arrived yet", k=3)

    assert result["is_relevant"] is True
    assert len(result["results"]) > 0
    assert len(result["results"]) <= 3
    assert all("text" in item for item in result["results"])
    assert all("document" in item for item in result["results"])
    assert all("section" in item for item in result["results"])
    assert all("score" in item for item in result["results"])


def test_off_topic_question_returns_no_context():
    result = retrieve_context("What is the weather in Paris?", k=3)

    assert result["is_relevant"] is False
    assert result["results"] == []


def test_empty_query_returns_no_context():
    result = retrieve_context("   ", k=3)

    assert result == {
        "is_relevant": False,
        "results": [],
    }


def test_invalid_k_raises_value_error():
    import pytest

    with pytest.raises(ValueError):
        retrieve_context("My package has not arrived yet", k=0)


from backend.app.rag.ingest import build_chunks

DOCS = {
    "delivery_policy.txt",
    "refund_policy.txt",
    "return_policy.txt",
    "damaged_products_policy.txt",
}


def test_chunks_have_metadata():
    chunks = build_chunks(1)
    assert len(chunks) == 34

    for chunk in chunks:
        assert chunk.metadata["document"] in DOCS
        assert chunk.metadata["section"].isdigit()
        assert len(chunk.page_content.split()) > 3


def test_all_documents_present():
    chunks = build_chunks(1)
    found = {chunk.metadata["document"] for chunk in chunks}
    assert found == DOCS


def test_group_size_2_has_fewer_chunks():
    assert len(build_chunks(2)) < len(build_chunks(1))


def test_no_duplicate_sections():
    chunks = build_chunks(1)
    keys = [
        (chunk.metadata["document"], chunk.metadata["section"])
        for chunk in chunks
    ]
    assert len(keys) == len(set(keys))

import pytest

from backend.app.rag.retriever import retrieve_context

EDGE = {
    "My package arrived but one of the parts is missing. What should I do?": {
        "damaged_products_policy.txt"
    },
    "I returned my order, but the return parcel was lost in transit. What should I do?": {
        "return_policy.txt",
        "delivery_policy.txt",
        "refund_policy.txt",
    },
}


@pytest.mark.parametrize("query,acceptable", EDGE.items())
def test_edge_case_retrieves_acceptable_docs(query, acceptable):
    result = retrieve_context(query, k=3)

    retrieved_documents = {
        item["document"]
        for item in result["results"]
    }

    assert retrieved_documents & acceptable
