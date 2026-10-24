
import pandas as pd

from backend.app.rag.retriever import _search

DATASET = "data/test_tickets.csv"
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

df = pd.read_csv(DATASET)

# Calculer les similarités une seule fois par ticket.
search_results = {}

for _, row in df.iterrows():
    search_results[row["ticket_id"]] = _search(
        str(row["ticket_text"]),
        k=3,
    )

in_scope = df[df["expected_document"].notna()]
out_scope = df[df["expected_document"].isna()]

metrics = []

for threshold in THRESHOLDS:
    hit1 = 0
    hit3 = 0
    abstentions = 0

    for _, row in in_scope.iterrows():
        results = search_results[row["ticket_id"]]

        filtered = [
            item for item in results
            if item["score"] >= threshold
        ]

        documents = [item["document"] for item in filtered]
        expected = row["expected_document"]

        if documents and documents[0] == expected:
            hit1 += 1

        if expected in documents:
            hit3 += 1

    for _, row in out_scope.iterrows():
        results = search_results[row["ticket_id"]]

        filtered = [
            item for item in results
            if item["score"] >= threshold
        ]

        if not filtered:
            abstentions += 1

    metrics.append({
        "threshold": threshold,
        "hit_at_1": round(100 * hit1 / len(in_scope), 2),
        "hit_at_3": round(100 * hit3 / len(in_scope), 2),
        "abstention": round(
            100 * abstentions / len(out_scope), 2
        ),
    })

results_df = pd.DataFrame(metrics)

print("\nCOMPARAISON DES SEUILS RAG")
print(results_df.to_string(index=False))

print("\nMeilleur Hit@3 :")
print(
    results_df.loc[
        results_df["hit_at_3"].idxmax()
    ].to_string()
)
