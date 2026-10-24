
import pandas as pd

from backend.app.rag.retriever import _search

tickets = pd.read_csv("data/test_tickets.csv")
predictions = pd.read_csv("outputs_classification_temp.csv")

df = tickets.merge(
    predictions[["ticket_id", "pred_category"]],
    on="ticket_id",
    how="left",
    validate="one_to_one",
)

if df["pred_category"].isna().any():
    raise ValueError("Certaines prédictions NLP sont manquantes.")

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

# Recherche vectorielle exécutée une seule fois par ticket.
raw_results = {
    row["ticket_id"]: _search(str(row["ticket_text"]), k=3)
    for _, row in df.iterrows()
}

rows = []

for threshold in thresholds:
    for approach in ["RAG seul", "NLP + RAG"]:
        hit1 = 0
        hit3 = 0
        abstentions = 0

        in_scope = df[df["expected_document"].notna()]
        out_scope = df[df["expected_document"].isna()]

        for _, ticket in df.iterrows():
            results = raw_results[ticket["ticket_id"]]

            if (
                approach == "NLP + RAG"
                and ticket["pred_category"] == "other"
            ):
                results = []
            else:
                results = [
                    item
                    for item in results
                    if item["score"] >= threshold
                ]

            docs = [item["document"] for item in results]
            expected = ticket["expected_document"]

            if pd.notna(expected):
                hit1 += int(bool(docs) and docs[0] == expected)
                hit3 += int(expected in docs)
            else:
                abstentions += int(not docs)

        rows.append({
            "approach": approach,
            "threshold": threshold,
            "hit_at_1": round(100 * hit1 / len(in_scope), 2),
            "hit_at_3": round(100 * hit3 / len(in_scope), 2),
            "abstention": round(
                100 * abstentions / len(out_scope), 2
            ),
        })

results_df = pd.DataFrame(rows)

print("\nCOMPARAISON RAG SEUL VS NLP + RAG")
print(results_df.to_string(index=False))
