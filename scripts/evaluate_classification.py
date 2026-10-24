"""Evaluate the LLM classifier on the held-out data/test_tickets.csv set.

Run from the project root: python -m scripts.evaluate_classification --limit 10
Remove --limit to evaluate all 40 tickets (uses API calls and may incur cost).
"""
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from backend.app.nlp.classification import analyze_ticket


ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional number of test tickets")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    data = pd.read_csv(ROOT / "data" / "test_tickets.csv").fillna("")
    if args.limit:
        data = data.head(args.limit)
    predictions = []
    for _, row in data.iterrows():
        pred = analyze_ticket(row["ticket_text"])
        predictions.append({
            "ticket_id": row["ticket_id"],
            "ticket_text": row["ticket_text"],
            "true_category": row["category"],
            "pred_category": pred["category"],
            "true_sentiment": row["sentiment"],
            "pred_sentiment": pred["sentiment"],
            "true_priority": row["priority"],
            "pred_priority": pred["priority"],
        })
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    result = pd.DataFrame(predictions)
    result.to_csv(out_dir / "classification_predictions.csv", index=False)
    print(f"Evaluated {len(result)} tickets")
    print("Category accuracy:", round(accuracy_score(result.true_category, result.pred_category), 4))
    print("Category macro-F1:", round(f1_score(result.true_category, result.pred_category, average="macro", zero_division=0), 4))
    print("Sentiment macro-F1:", round(f1_score(result.true_sentiment, result.pred_sentiment, average="macro", zero_division=0), 4))
    print("\nCategory report:\n", classification_report(result.true_category, result.pred_category, zero_division=0))
    labels = sorted(set(result.true_category) | set(result.pred_category))
    pd.DataFrame(confusion_matrix(result.true_category, result.pred_category, labels=labels),
                 index=labels, columns=labels).to_csv(out_dir / "category_confusion_matrix.csv")
    errors = result[result.true_category != result.pred_category].head(5)
    print("\nUp to five category errors:\n", errors[["ticket_id", "ticket_text", "true_category", "pred_category"]].to_string(index=False))
    print(f"\nFiles saved in: {out_dir}")

if __name__ == "__main__":
    main()
