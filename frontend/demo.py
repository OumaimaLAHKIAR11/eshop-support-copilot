"""Simulated responses for UI development (DEMO_MODE=true only).

Nothing here calls an LLM, a RAG pipeline, or the backend. Replies are canned,
keyword-matched, and deliberately generic: they never state order status,
tracking details, delivery dates, refund approvals, or completed returns.
The return value mirrors the real /process-ticket response shape.
"""

import os
import time

_DELIVERY = (
    "I'm sorry your order is taking longer than expected. I can't see your order "
    "or live tracking from here, so I can't confirm where it is or when it will arrive.\n\n"
    "Here is what usually helps:\n"
    "1. Check your order confirmation or shipping email for a tracking link.\n"
    "2. If tracking hasn't updated, or the delivery window you were given has passed, "
    "contact our support team with your order number so they can look into it."
)

_RETURNS = (
    "Happy to help with a return. I can't check your specific order, so I can't confirm "
    "whether an item is eligible. Eligible items and time limits are set by the shop's returns policy.\n\n"
    "To get started:\n"
    "1. Have your order number ready.\n"
    "2. Look for a return option in your order details, or ask our support team to start one.\n"
    "3. Keep the item unused and in its original packaging where possible.\n\n"
    "A return is only complete once the shop has confirmed it received and processed the item."
)

_REFUNDS = (
    "Refunds depend on the shop's refund policy and on how your return or cancellation is "
    "handled, so I can't promise approval or an amount. A refund is often considered after a "
    "returned item has been received and checked, and how long it takes to show up can vary "
    "by payment method.\n\n"
    "If you have already sent something back and haven't heard anything, share your order "
    "number with our support team and they can check where things stand."
)

_FALLBACK = (
    "I can help with questions about delivery, returns, and refunds. I can't access your "
    "account or order details here, so for anything specific to your order, please contact "
    "our support team.\n\nCould you tell me a little more about what you need?"
)

_RULES = (
    (("refund", "money back", "reimburse", "charged"), "refund_question", _REFUNDS, False),
    (("return", "exchange", "send back", "send it back"), "return_question", _RETURNS, False),
    (("deliver", "delay", "late", "track", "shipping", "arrive", "where is my"), "delivery_issue", _DELIVERY, True),
)


def simulate_ticket(ticket_id: str, message: str, language: str = "en") -> dict:
    time.sleep(float(os.getenv("DEMO_DELAY", "0.9")))  # lets the loading state be seen
    text = message.lower()
    for keywords, category, reply, review in _RULES:
        if any(k in text for k in keywords):
            return _payload(ticket_id, category, reply, review)
    return _payload(ticket_id, "general", _FALLBACK, False)


def _payload(ticket_id: str, category: str, reply: str, review: bool) -> dict:
    return {
        "ticket_id": ticket_id,
        "category": category,
        "sentiment": "neutral",
        "priority": "normal",
        "suggested_reply": reply,
        "sources": [],
        "needs_human_review": review,
    }