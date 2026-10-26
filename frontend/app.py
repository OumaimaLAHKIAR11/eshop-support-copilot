"""ShopAssist - E-Shop Customer Support Copilot (Streamlit frontend).

Run from the repository root:  streamlit run frontend/app.py
Set DEMO_MODE=true to preview the UI with simulated replies (no backend needed).

The last ten turns of the current chat are sent as context with each request.
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

from frontend.reply_format import customer_reply_text
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

BRAND_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="220" height="48" viewBox="0 0 220 48" role="img" aria-label="ShopAssist">
<rect x="0" y="4" width="40" height="40" rx="12" fill="#0e7664"/>
<g fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M10 21h20l-1.6 17H11.6L10 21Z"/><path d="M15 22v-4a5 5 0 0 1 10 0v4"/>
</g><text x="54" y="32" fill="#17212b" font-family="Segoe UI,Arial,sans-serif" font-size="26" font-weight="650" letter-spacing="-.7">ShopAssist</text></svg>"""

TYPING_HTML = '<div class="typing" role="status" aria-label="Assistant is typing"><span></span><span></span><span></span></div>'

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
def get_reply(message: str, history: list | None = None) -> dict:
    ss = st.session_state
    ss.ticket_counter += 1
    ticket_id = f"CHAT-{ss.session_id}-{ss.ticket_counter:03d}"
    if demo_mode():
        return simulate_ticket(ticket_id, message)
    return process_ticket(ticket_id=ticket_id, message=message,
                          language=ss.get("reply_language", "en"), history=history)


def answer(chat: dict, prompt: str) -> None:
    """Fetch a reply, store it (or an error state) on the chat, then rerun."""
    with st.container(key="msg_assistant_pending"), st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        slot = st.empty()
        slot.markdown(TYPING_HTML, unsafe_allow_html=True)
        try:
            history = [{"role": m["role"], "content": m["content"][:6000]}
                       for m in chat["messages"][:-1][-10:]]
            result = get_reply(prompt, history)
            reply = str(result.get("suggested_reply") or "").strip()
            if not reply:
                raise ValueError("empty reply")
            chat["messages"].append({
                "role": "assistant",
                "content": reply,
                "needs_review": bool(result.get("needs_human_review", False)),
                "sources": result.get("sources", []),
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
        st.button("New conversation", icon=":material/edit_square:", type="primary", key="new_chat", on_click=new_chat, use_container_width=True)

        chats = [(cid, c) for cid, c in ss.chats.items() if c["messages"]]
        st.markdown('<div class="side-label">Conversations</div>', unsafe_allow_html=True)
        if chats:
            with st.container(key="chatlist"):
                for cid, c in reversed(chats):
                    is_current = cid == ss.current_id
                    st.button(
                        c["title"],
                        key=f"{'chatcur' if is_current else 'chat'}_{cid}",
                        on_click=open_chat, args=(cid,), type="tertiary", icon=":material/chat_bubble_outline:",
                        use_container_width=True,
                    )

        else:
            st.markdown('<p class="side-empty">Your conversations will appear here.</p>', unsafe_allow_html=True)


def render_welcome() -> None:
    st.markdown(
        """<div class="welcome">
        <span class="welcome-label">CUSTOMER SUPPORT</span>
        <h1>How can we help?</h1>
        <p>Get help with a delivery, a return, or a refund.</p></div>""",
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
            st.markdown(customer_reply_text(msg["content"], msg.get("sources", [])))
            if role == "assistant":
                if msg.get("demo"):
                    st.caption("demo mode")


def render_error(kind: str) -> None:
    title, body = ERRORS[kind]
    with st.container(key="error_box"):
        st.markdown(f"**{title}**  \n{body}")
        st.button("Try again", icon=":material/refresh:", key="retry_btn", on_click=request_retry)


def main() -> None:
    st.set_page_config(page_title="ShopAssist", page_icon=":material/storefront:", layout="centered", initial_sidebar_state="expanded")
    init_state()
    st.logo(BRAND_SVG, size="large", icon_image=BRAND_SVG)

    helper = ("Demo mode: replies are simulated. " if demo_mode() else "") + "Help with deliveries, returns and refunds."
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
