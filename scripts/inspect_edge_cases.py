
from backend.app.rag.retriever import _search, retrieve_context

CASES = {
    "T112": {
        "label": "damaged_product",
        "query": (
            "My package arrived but one of the parts "
            "is missing. What should I do?"
        ),
    },
    "T268": {
        "label": "return_request",
        "query": (
            "I returned my order, but the return parcel "
            "was lost in transit. What should I do?"
        ),
    },
}

for ticket_id, case in CASES.items():
    query = case["query"]

    print(f"\n{'=' * 65}")
    print(f"Ticket : {ticket_id}")
    print(f"Label de référence : {case['label']}")
    print(f"Question : {query}")

    print("\nTOP 8 — Résultats bruts")
    results = _search(query, k=8)

    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. Score : {result['score']:.4f}")
        print(f"Document : {result['document']}")
        print(f"Section : {result['section']}")
        print(f"Texte : {result['text']}")

    print("\nRETRIEVAL AVEC SEUIL")
    filtered = retrieve_context(query, k=8)

    print("is_relevant :", filtered["is_relevant"])
    print("Nombre de résultats :", len(filtered["results"]))
