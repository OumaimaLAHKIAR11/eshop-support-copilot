import os

import requests

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


def process_ticket(ticket_id: str, message: str, language: str = "en", history: list | None = None) -> dict:
    """POST /api/v1/process-ticket with optional bounded conversation history.

    Raises requests.RequestException on network/HTTP errors and ValueError
    if the body is not a JSON object.
    """
    response = requests.post(
        f"{API_BASE_URL}/api/v1/process-ticket",
        json={
            "ticket_id": ticket_id,
            "message": message,
            "language": language,
            "history": history or [],
        },
        timeout=150,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response shape")
    return data
