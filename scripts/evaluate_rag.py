"""Evaluate document retrieval and exact rule coverage, including compound tickets."""
import argparse
import csv
import json
from pathlib import Path

from backend.app.rag.retriever import retrieve_context


def source_ids(results):
    found = set()
    for result in results:
        section = result['section']
        if '-' in section:
            first, last = map(int, section.split('-'))
            sections = map(str, range(first, last + 1))
        else:
            sections = [section]
        found.update(f"{result['document']}:{section}" for section in sections)
    return found


def evaluate(rows, retrieve=retrieve_context, k=3):
    records = []
    for row in rows:
        results = retrieve(row['ticket_text'], k=k)['results']
        expected = set(filter(None, row.get('required_sources', '').split('|')))
        found = source_ids(results)
        records.append({
            'ticket_id': row['ticket_id'],
            'document_hit': row.get('expected_document') in [r['document'] for r in results]
                            if row.get('expected_document') else not results,
            'required_sources': sorted(expected), 'retrieved_sources': sorted(found),
            'section_recall': len(expected & found) / len(expected) if expected else None,
            'all_required_sections_found': expected <= found if expected else None,
        })
    annotated = [r for r in records if r['required_sources']]
    return {'tickets': len(records), 'k': k,
            'document_accuracy': sum(r['document_hit'] for r in records) / len(records) if records else None,
            'section_annotated_tickets': len(annotated),
            'mean_section_recall': sum(r['section_recall'] for r in annotated) / len(annotated) if annotated else None,
            'full_coverage_rate': sum(r['all_required_sections_found'] for r in annotated) / len(annotated) if annotated else None,
            'results': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/test_tickets.csv')
    parser.add_argument('--output', default='outputs/rag_evaluation.json')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--multi', action='store_true', help='Evaluate the full multi-question search')
    args = parser.parse_args()
    retrieve = retrieve_context
    if args.multi:
        from backend.app.api.routes import ticket_context
        from backend.app.schemas import TicketRequest
        retrieve = lambda text, k: ticket_context(TicketRequest(ticket_id='eval', message=text))
    with open(args.dataset, encoding='utf-8-sig', newline='') as stream:
        report = evaluate(list(csv.DictReader(stream)), retrieve=retrieve, k=args.k)
        report['mode'] = 'multi-question with neighboring policy rules' if args.multi else 'single-query'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
