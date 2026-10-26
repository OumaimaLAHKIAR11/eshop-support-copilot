"""Run the reported three-turn conversation through the real configured services.

This script calls Azure and uses the local policy index. No services are mocked.
"""
import json
from fastapi.testclient import TestClient
from backend.app.main import app

MESSAGES = [
    'Thank you for your excellent service! My order arrived perfectly',
    'This is unacceptable! My package is two weeks late and nobody is helping me',
    'I would like to know how to return an item.',
]


def main():
    import os
    if os.getenv("TRACE_SUPPORT_DRAFTS") == "1":
        from backend.app.generation import generation
        from langchain_core.runnables import RunnableLambda
        factory = generation._get_generator
        def traced(prompt):
            reply = factory().invoke(prompt)
            print("GENERATION_DRAFT", reply.model_dump() if hasattr(reply, "model_dump") else reply, flush=True)
            return reply
        generation._get_generator = lambda: RunnableLambda(traced)
    results = []
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-then-delay', action='store_true')
    parser.add_argument('--reported-damage', action='store_true')
    parser.add_argument('--case-followup', action='store_true')
    parser.add_argument('--offline-embeddings', action='store_true', help='Use the already cached embedding model')
    args = parser.parse_args()
    if args.offline_embeddings:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    messages = MESSAGES[2:] + MESSAGES[1:2] if args.return_then_delay else MESSAGES
    if args.reported_damage:
        messages = [MESSAGES[2], MESSAGES[1], 'My laptop arrived with a cracked screen']
    if args.case_followup:
        messages = ['My laptop arrived with a cracked screen.', 'My order number is NS-12345', 'What else do you need from me?']
    history = []
    client = TestClient(app)
    for index, message in enumerate(messages):
        response = client.post('/api/v1/process-ticket', json={
            'ticket_id': f'LIVE-CHECK-{index}', 'message': message,
            'language': 'en', 'history': history[-10:],
        })
        data = response.json()
        results.append((response.status_code, data))
        print(json.dumps({'message': message, 'status': response.status_code, 'result': data}, ensure_ascii=True), flush=True)
        history.append({'role': 'user', 'content': message})
        if response.status_code == 200:
            history.append({'role': 'assistant', 'content': data['suggested_reply']})

    assert all(status == 200 for status, _ in results), "An API request failed"
    for message, (_, data) in zip(messages, results):
        if args.case_followup:
            text = data['suggested_reply'].lower()
            assert data['category'] == 'damaged_product', 'The damaged-product case was lost'
            assert 'photo' in text, 'The missing evidence was not requested'
            assert 'reliable answer' not in text, 'Generic refusal in follow-up'
            if message == messages[1]:
                assert 'NS-12345' in data['suggested_reply'], 'New order number was not acknowledged'
            if message == messages[2]:
                assert 'already provided your order number' in text, 'Customer-provided order number was forgotten'
                assert 'could you share your order number' not in text, 'Repeated request for the supplied number'
        elif message == MESSAGES[0]:
            assert not data['needs_human_review'], 'Praise was unnecessarily escalated'
        elif message == MESSAGES[1]:
            assert 'order number' in data['suggested_reply'].lower(), 'Delayed delivery has no request for the missing order number'
            assert data['needs_human_review'], 'Shipment verification should be flagged'
        elif message == 'My laptop arrived with a cracked screen':
            text = data['suggested_reply'].lower()
            assert 'order number' in text and 'photo' in text, 'Damage response lacks concrete next steps'
            assert 'before providing a reliable answer' not in text, 'Generic damage response'
            assert data['needs_human_review'], 'Damage approval requires support review'
            assert any(s['document'] == 'damaged_products_policy.txt' for s in data['sources']), 'Damage policy was not retrieved'
        elif message == MESSAGES[2]:
            assert not data['needs_human_review'], 'General return guidance was unnecessarily escalated'
            assert len(data['suggested_reply'].split()) < 120, 'Return guidance is too verbose'
            assert any(source['document'] == 'return_policy.txt' for source in data['sources']), 'Missing policy source'
    if args.case_followup:
        assert results[1][1]['suggested_reply'] != results[2][1]['suggested_reply'], 'Conversation is repeating the same reply'
    print('Live conversation regression: PASS', flush=True)


if __name__ == '__main__':
    main()
