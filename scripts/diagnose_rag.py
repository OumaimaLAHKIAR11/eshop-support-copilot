
import pandas as pd

from backend.app.rag.retriever import _search, retrieve_context

df = pd.read_csv("data/test_tickets.csv")

for _, row in df.iterrows():
    result = retrieve_context(str(row["ticket_text"]), k=3)

    expected = row["expected_document"]
    docs = [item["document"] for item in result["results"]]

    failed_in_scope = pd.notna(expected) and expected not in docs
    accepted_out_scope = pd.isna(expected) and result["is_relevant"]

    if not (failed_in_scope or accepted_out_scope):
        continue

    print("\n" + "=" * 65)
    print("Ticket :", row["ticket_id"])
    print("Texte :", row["ticket_text"])
    print("Document attendu :", expected)
    print("RAG pertinent :", result["is_relevant"])

    # Recherche brute avant filtrage par le seuil
    raw_results = _search(str(row["ticket_text"]), k=5)

    for item in raw_results:
        print(
            f"Score={item['score']:.4f} | "
            f"Document={item['document']} | "
            f"Section={item['section']}"
        )
