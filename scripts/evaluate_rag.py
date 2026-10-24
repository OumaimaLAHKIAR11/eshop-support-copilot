
import pandas as pd

from backend.app.rag.retriever import retrieve_context

df = pd.read_csv("data/test_tickets.csv")

in_scope = df[df["expected_document"].notna()]
out_scope = df[df["expected_document"].isna()]

hit1 = 0
hit3 = 0
abstentions = 0
errors = []

for _, row in in_scope.iterrows():
    result = retrieve_context(str(row["ticket_text"]), k=3)
    docs = [item["document"] for item in result["results"]]
    expected = row["expected_document"]

    hit1 += bool(docs and docs[0] == expected)
    hit3 += expected in docs

    if expected not in docs:
        errors.append({
            "ticket_id": row["ticket_id"],
            "expected": expected,
            "retrieved": docs,
        })

for _, row in out_scope.iterrows():
    result = retrieve_context(str(row["ticket_text"]), k=3)

    if not result["is_relevant"]:
        abstentions += 1

print("Tickets avec politique :", len(in_scope))
print("Tickets sans politique :", len(out_scope))

print("Hit@1 :", round(100 * hit1 / len(in_scope), 2), "%")
print("Hit@3 :", round(100 * hit3 / len(in_scope), 2), "%")
print(
    "Taux d'abstention :",
    round(100 * abstentions / len(out_scope), 2),
    "%"
)

print("\nErreurs de retrieval :")
for error in errors:
    print(error)
