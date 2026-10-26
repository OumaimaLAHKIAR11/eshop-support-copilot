from scripts.compare_thresholds import metrics, split_rows


def test_split_keeps_duplicate_text_together():
    rows = [{"ticket_text": text, "expected_document": document}
            for document in ("policy.txt", "") for text in ("A", "B", "C", " A ")]
    calibration, validation = split_rows(rows)
    assert calibration and validation
    assert not ({r["ticket_text"].strip().casefold() for r in calibration}
                & {r["ticket_text"].strip().casefold() for r in validation})


def test_metrics_balance_recall_and_abstention():
    rows = [{"ticket_text": "return", "expected_document": "return.txt"},
            {"ticket_text": "weather", "expected_document": ""}]
    hits = {"return": [{"document": "return.txt", "score": 0.7}],
            "weather": [{"document": "return.txt", "score": 0.2}]}
    assert metrics(rows, hits, 0.35)["balanced_score"] == 1
    assert metrics(rows, hits, 0.8)["hit_at_3"] == 0
