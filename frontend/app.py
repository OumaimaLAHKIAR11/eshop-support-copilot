"""ShopAssist - E-Shop Customer Support Copilot (Streamlit frontend).

Run from the repository root:  streamlit run frontend/app.py
Set DEMO_MODE=true to preview the UI with simulated replies (no backend needed).

Limitation: POST /api/v1/process-ticket handles ONE message per request and has no
conversation ID or history field, so the assistant has no memory of earlier turns.
History shown here is only a per-browser-session display convenience.
"""

import logging
import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.api_client import process_ticket  # noqa: E402
from frontend.demo import simulate_ticket  # noqa: E402

log = logging.getLogger("shopassist")
CSS_PATH = Path(__file__).with_name("styles.css")

ASSISTANT_AVATAR = ":material/support_agent:"
USER_AVATAR = ":material/person:"
SUGGESTIONS = [
    ("My order is delayed", ":material/local_shipping:"),
    ("What is your refund policy?", ":material/payments:"),
    ("How can I return a product?", ":material/assignment_return:"),
]

BRAND_HTML = """
<div class="brand">
  <span class="brand-mark">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M6 7h12l-1 11a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L6 7z"/><path d="M9 7a3 3 0 0 1 6 0"/>
    </svg>
  </span>
  <span><span class="brand-name">ShopAssist</span><br><span class="brand-sub">E-Shop Customer Support</span></span>
</div>
"""

TYPING_HTML = '<div class="typing" role="status" aria-label="Assistant is typing"><span></span><span></span><span></span></div>'
NOTICE_HTML = '<div class="notice">This request may need a member of our support team to review it.</div>'

ERRORS = {
    "unavailable": ("We couldn't reach the assistant.", "The service isn't responding right now. Please try again in a moment."),
    "invalid": ("Something went wrong on our side.", "We received an unexpected answer. Please try again."),
}


# ---------- state ----------
def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:8])
    ss.setdefault("chats", {})  # id -> {"title", "messages", "error"}
    ss.setdefault("current_id", uuid.uuid4().hex)
    ss.setdefault("ticket_counter", 0)
    ss.setdefault("backend_ok", None)  # None = not contacted yet
    ss.setdefault("pending", None)
    ss.setdefault("retry", False)


def current_chat() -> dict:
    ss = st.session_state
    return ss.chats.setdefault(ss.current_id, {"title": "New chat", "messages": [], "error": None})


def new_chat() -> None:
    ss = st.session_state
    if current_chat()["messages"]:  # an empty chat is already a "new chat"
        ss.current_id = uuid.uuid4().hex
    ss.pending, ss.retry = None, False


def open_chat(chat_id: str) -> None:
    st.session_state.current_id = chat_id
    st.session_state.pending = None


def set_pending(text: str) -> None:
    st.session_state.pending = text


def request_retry() -> None:
    st.session_state.retry = True


def shorten(text: str, n: int = 38) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------- backend ----------
def get_reply(message: str) -> dict:
    ss = st.session_state
    ss.ticket_counter += 1
    ticket_id = f"CHAT-{ss.session_id}-{ss.ticket_counter:03d}"
    if demo_mode():
        return simulate_ticket(ticket_id, message)
    return process_ticket(ticket_id=ticket_id, message=message)


def answer(chat: dict, prompt: str) -> None:
    """Fetch a reply, store it (or an error state) on the chat, then rerun."""
    with st.container(key="msg_assistant_pending"), st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        slot = st.empty()
        slot.markdown(TYPING_HTML, unsafe_allow_html=True)
        try:
            result = get_reply(prompt)
            reply = str(result.get("suggested_reply") or "").strip()
            if not reply:
                raise ValueError("empty reply")
            chat["messages"].append({
                "role": "assistant",
                "content": reply,
                "needs_review": bool(result.get("needs_human_review", False)),
                "demo": demo_mode(),
            })
            chat["error"] = None
            st.session_state.backend_ok = True
        except requests.RequestException as exc:
            log.warning("Backend request failed: %s", exc)
            if isinstance(exc, requests.HTTPError):
                chat["error"] = "invalid"
            else:
                chat["error"] = "unavailable"
            st.session_state.backend_ok = False
        except (ValueError, TypeError, AttributeError) as exc:
            log.warning("Invalid backend response: %s", exc)
            chat["error"] = "invalid"
        slot.empty()
    st.rerun()


# ---------- rendering ----------
def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown(BRAND_HTML, unsafe_allow_html=True)
        st.button("New chat", icon=":material/add:", type="primary", key="new_chat", on_click=new_chat)

        chats = [(cid, c) for cid, c in ss.chats.items() if c["messages"]]
        if chats:
            st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
            with st.container(key="chatlist"):
                for cid, c in reversed(chats):
                    is_current = cid == ss.current_id
                    st.button(
                        c["title"],
                        key=f"{'chatcur' if is_current else 'chat'}_{cid}",
                        on_click=open_chat, args=(cid,), type="tertiary",
                    )

        st.markdown('<div class="side-spacer"></div>', unsafe_allow_html=True)
        if demo_mode():
            st.markdown('<span class="pill pill-demo">Demo mode</span><div class="side-note">.</div>', unsafe_allow_html=True)
        else:
            state = {None: ("idle", "Live mode"), True: ("ok", "Assistant reachable"), False: ("down", "Assistant unreachable")}[ss.backend_ok]
            st.markdown(f'<span class="pill pill-{state[0]}"><i></i>{state[1]}</span>', unsafe_allow_html=True)


def render_welcome() -> None:
    st.markdown(
        """<div class="welcome">
        <h1>How can we help you today?</h1>
        <p>Ask anything about your delivery, returns, or refunds.</p></div>""",
        unsafe_allow_html=True,
    )
    with st.container(key="suggestions"):
        for col, (text, icon) in zip(st.columns(3), SUGGESTIONS):
            with col:
                st.button(text, icon=icon, key=f"sugg_{text}", on_click=set_pending, args=(text,))


def render_message(msg: dict, index: int) -> None:
    """Le rôle (user / assistant) est encodé dans la clé du conteneur -> classe CSS
    `st-key-msg_<role>_<index>` ; l'alignement est donc piloté par le rôle, pas par la position."""
    role = "user" if msg["role"] == "user" else "assistant"
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
    with st.container(key=f"msg_{role}_{index}"):
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if role == "assistant":
                if msg.get("needs_review"):
                    st.markdown(NOTICE_HTML, unsafe_allow_html=True)
                if msg.get("demo"):
                    st.caption("demo mode")


def render_error(kind: str) -> None:
    title, body = ERRORS[kind]
    with st.container(key="error_box"):
        st.markdown(f"**{title}**  \n{body}")
        st.button("Try again", icon=":material/refresh:", key="retry_btn", on_click=request_retry)


def main() -> None:
    st.set_page_config(page_title="ShopAssist", page_icon=":material/storefront:", layout="centered", initial_sidebar_state="auto")
    init_state()

    helper = ("Demo mode: replies are simulated. " if demo_mode() else "") + "ShopAssist specializes in e-commerce support: delivery, returns and refunds."
    css = CSS_PATH.read_text(encoding="utf-8")  # @import doit rester en tête de sa feuille : blocs <style> séparés
    st.markdown(f"<style>\n{css}\n</style>\n<style>:root{{--helper-text:\"{helper}\";}}</style>", unsafe_allow_html=True)

    render_sidebar()
    chat = current_chat()
    msgs = chat["messages"]

    typed = st.chat_input("Message ShopAssist")
    prompt = (typed or st.session_state.pending or "").strip()
    st.session_state.pending = None

    to_answer = None
    if prompt:
        msgs.append({"role": "user", "content": prompt})
        if chat["title"] == "New chat":
            chat["title"] = shorten(prompt)
        chat["error"] = None
        to_answer = prompt
    elif st.session_state.retry:
        st.session_state.retry = False
        if msgs and msgs[-1]["role"] == "user" and chat["error"]:
            chat["error"] = None
            to_answer = msgs[-1]["content"]

    if not msgs:
        render_welcome()
        return

    for i, m in enumerate(msgs):
        render_message(m, i)
    if to_answer:
        answer(chat, to_answer)
    elif chat["error"]:
        render_error(chat["error"])


main()