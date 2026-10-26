from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_chat_has_no_support_preparation_form_even_with_old_session_state(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', 'false')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'frontend/app.py', default_timeout=20).run()
    chat_id = app.session_state['current_id']
    app.session_state['chats'] = {chat_id: {
        'title': 'Damaged laptop', 'error': None,
        'messages': [{'role': 'assistant', 'content': 'Please have photos ready for support.'}],
        'support_draft': {'issue': 'Old draft', 'order_number': 'NS-12345'},
        'support_case': {'reference': 'SUP-OLD', 'download_text': 'Old saved request'}}}
    app.run()
    assert not app.exception
    assert not app.text_area
    assert not app.form
    assert not any('support request' in button.label.lower() for button in app.button)
    assert any('Please have photos ready' in element.value for element in app.markdown)

def test_professional_sidebar_has_no_language_or_connection_indicators(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', 'false')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'frontend/app.py', default_timeout=20).run()
    assert not app.exception
    assert not app.sidebar.selectbox
    assert any(button.label == 'New conversation' for button in app.sidebar.button)
    text = '\n'.join(element.value for element in app.markdown)
    for unwanted in ['Reply language', 'Assistant reachable', 'Assistant unreachable', 'Live mode']:
        assert unwanted not in text
    assert 'Conversations' in text


def test_new_conversation_and_history_navigation_still_work(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', 'false')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'frontend/app.py', default_timeout=20).run()
    old_id = app.session_state['current_id']
    app.session_state['chats'] = {old_id: {'title': 'A damaged laptop', 'error': None,
        'messages': [{'role': 'user', 'content': 'My laptop arrived damaged.'},
                     {'role': 'assistant', 'content': 'Please share your order number.'}]}}
    app.run()
    app.button(key='new_chat').click().run()
    assert not app.exception
    assert app.session_state['current_id'] != old_id
    assert any('How can we help?' in element.value for element in app.markdown)
    next(button for button in app.sidebar.button if button.label == 'A damaged laptop').click().run()
    assert not app.exception
    assert app.session_state['current_id'] == old_id
    assert any('Please share your order number' in element.value for element in app.markdown)


def test_native_logo_is_set_for_open_and_closed_sidebar(monkeypatch):
    import streamlit as st
    from unittest.mock import Mock
    logo = Mock(wraps=st.logo)
    monkeypatch.setattr(st, 'logo', logo)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'frontend/app.py', default_timeout=20).run()
    assert not app.exception
    assert logo.called
    image = logo.call_args.args[0]
    assert 'ShopAssist' in image
    assert logo.call_args.kwargs['icon_image'] == image
