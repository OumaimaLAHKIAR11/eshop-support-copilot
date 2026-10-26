"""Resolve a support case from customer turns; assistant text is never case evidence."""
import re
from dataclasses import dataclass

ORDER_UPDATE = re.compile(
    r"^(?:(?:my|our|the|mon|notre|here is my)\s+)?(?:order\s*(?:number|no\.?|#)|num[eé]ro\s+de\s+commande)"
    r"\s*(?:(?:is|est|:|#)\s*)?(?=[A-Z0-9-]*\d)[A-Z0-9-]+[.! ]*$",
    re.IGNORECASE,
)
BARE_ID = re.compile(r"^\s*(?:NS-\d+|(?=[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,})\s*[.!]?\s*$", re.IGNORECASE)
FOLLOWUP_QUESTION = re.compile(
    r"\bwhat (?:else|more)\b|\bwhat.*(?:need from me|do you need|next|prepare)\b"
    r"|\b(?:autre chose|besoin de moi|ensuite|que dois.je pr[eé]parer)\b"
    r"|^(?:and|et|what about|how long|combien)\b",
    re.IGNORECASE,
)
SUBJECT = re.compile(r"\b(?:laptop|screen|phone|jacket|parcel|package|order|item|product|delivery|refund|colis|commande|article|produit|livraison|remboursement)\b", re.IGNORECASE)
REFERENCE = re.compile(r"\b(?:it|that|this|they|them|those|cela|ceci)\b", re.IGNORECASE)
DETAIL_UPDATE = re.compile(r"^(?:I (?:have |already )?(?:sent|attached|uploaded)|here (?:is|are)|j.ai (?:envoy[eé]|joint))\b", re.IGNORECASE)


@dataclass(frozen=True)
class CaseContext:
    issue: str | None
    updates: tuple[str, ...]
    is_followup: bool
    text: str


def is_case_followup(message: str) -> bool:
    text = message.strip()
    if ORDER_UPDATE.fullmatch(text) or BARE_ID.fullmatch(text):
        return True
    photo_update = re.search(
        r"^I (?:have|took|prepared|don't have).*\b(?:photos?|pictures?|images?)\b"
        r"|^j.ai.*\bphotos?\b", text, re.IGNORECASE)
    if FOLLOWUP_QUESTION.search(text) or DETAIL_UPDATE.search(text) or photo_update:
        return True
    return bool(REFERENCE.search(text) and not SUBJECT.search(text))


def resolve_case(message: str, history: list[dict]) -> CaseContext:
    customers = [turn['content'] for turn in history if turn.get('role') == 'user']
    if not customers or not is_case_followup(message):
        return CaseContext(None, (), False, message)
    # Walk back through details-only turns to the most recent actual issue.
    anchor = len(customers) - 1
    while anchor >= 0 and is_case_followup(customers[anchor]):
        anchor -= 1
    if anchor < 0:
        return CaseContext(None, (), False, message)
    issue = customers[anchor]
    updates = tuple(customers[anchor + 1:])
    text = ('Active customer issue (still the subject of this follow-up):\n' + issue
            + '\nCustomer-provided updates:\n' + '\n'.join(updates)
            + '\nCurrent customer message:\n' + message)
    return CaseContext(issue, updates, True, text)
