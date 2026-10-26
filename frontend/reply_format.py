"""Keep detailed policy IDs in the source list rather than in customer prose."""
import re


def customer_reply_text(content: str, sources: list[dict]) -> str:
    ids = {f"{source['document']}:{source['section']}" for source in sources}
    # Remove only verified source markers; preserve customer text and other brackets.
    return re.sub(r"[ \t]*\[([^\[\]\n]+)\]", lambda match: '' if match.group(1) in ids else match.group(0), content)


def source_label(source: dict) -> str:
    title = source['document'].removesuffix('.txt').replace('_', ' ').capitalize()
    return f"{title} — rule {source['section']}"
