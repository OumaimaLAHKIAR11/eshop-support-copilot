"""Calibrate on a deterministic split and report held-out retrieval metrics.

Run: python -m scripts.compare_thresholds --dataset data/test_tickets.csv
The recommendation is reported only; production configuration is never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from backend.app.rag.retriever import _search

THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def split_rows(rows):
    calibration, validation = [], []
    # Keep repeated ticket text in the same split; stratify by expected document.
    groups = {}
    for row in rows:
        groups.setdefault(row.get("expected_document", ""), {}).setdefault(
            row["ticket_text"].strip().casefold(), []).append(row)
    for group in groups.values():
        texts = sorted(group, key=lambda text: hashlib.sha256(text.encode()).hexdigest())
        for i, text in enumerate(texts):
            (validation if i % 3 == 0 else calibration).extend(group[text])
    return calibration, validation


def metrics(rows, hits, threshold):
    inside = [row for row in rows if row.get("expected_document")]
    outside = [row for row in rows if not row.get("expected_document")]
    def docs(row):
        return [x["document"] for x in hits[row["ticket_text"]] if x["score"] >= threshold]
    hit1 = sum(bool(docs(r)) and docs(r)[0] == r["expected_document"] for r in inside)
    hit3 = sum(r["expected_document"] in docs(r) for r in inside)
    abstain = sum(not docs(r) for r in outside)
    recall = hit3 / len(inside) if inside else 0
    specificity = abstain / len(outside) if outside else 0
    return {"threshold": threshold, "in_scope": len(inside), "out_of_scope": len(outside),
            "hit_at_1": hit1 / len(inside) if inside else None,
            "hit_at_3": recall if inside else None,
            "abstention": specificity if outside else None,
            "balanced_score": (recall + specificity) / 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/test_tickets.csv")
    parser.add_argument("--output", default="outputs/rag_threshold_evaluation.json")
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    calibration, validation = split_rows(rows)
    if not all(any(r.get("expected_document") for r in split) and
               any(not r.get("expected_document") for r in split)
               for split in (calibration, validation)):
        raise ValueError("Both splits require in-scope and out-of-scope examples.")
    hits = {text: _search(text, k=3) for text in dict.fromkeys(r["ticket_text"] for r in rows)}
    comparison = [metrics(calibration, hits, threshold) for threshold in THRESHOLDS]
    best = max(comparison, key=lambda result: (result["balanced_score"], result["threshold"]))
    report = {"dataset": args.dataset, "calibration": comparison,
              "recommended_threshold": best["threshold"],
              "validation": metrics(validation, hits, best["threshold"]),
              "note": "Split by normalized ticket text. Validation was not used for selection. No configuration changed."}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
