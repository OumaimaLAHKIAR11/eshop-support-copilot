from scripts.evaluate_rag import evaluate, source_ids


def test_grouped_sections_expand():
    assert source_ids([{'document': 'a', 'section': '1-2'}]) == {'a:1', 'a:2'}


def test_document_hit_does_not_hide_missing_rules():
    rows = [{'ticket_id': '1', 'ticket_text': 'compound', 'expected_document': 'a',
             'required_sources': 'a:1|b:2'}]
    report = evaluate(rows, lambda text, k: {'results': [{'document': 'a', 'section': '1'}]})
    assert report['document_accuracy'] == 1
    assert report['mean_section_recall'] == 0.5
    assert report['full_coverage_rate'] == 0
