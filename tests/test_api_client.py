from unittest.mock import Mock
from frontend import api_client


def test_client_forwards_history_and_language(monkeypatch):
    response = Mock()
    response.json.return_value = {'suggested_reply': 'Bonjour'}
    post = Mock(return_value=response)
    monkeypatch.setattr(api_client.requests, 'post', post)
    history = [{'role': 'user', 'content': 'Returns?'}]
    api_client.process_ticket('T1', 'And timing?', 'fr', history)
    assert post.call_args.kwargs['json']['history'] == history
    assert post.call_args.kwargs['json']['language'] == 'fr'
    response.raise_for_status.assert_called_once()
