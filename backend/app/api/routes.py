
from fastapi import APIRouter, HTTPException

from backend.app.schemas import (
    HealthResponse,
    TicketRequest,
    TicketResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check whether the API is running."""
    return {"status": "ok"}


@router.post("/api/v1/analyze")
def analyze_ticket_endpoint(request: TicketRequest):
    """
    Temporary endpoint.
    Replace the simulated result with Member 1's function.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Ticket message cannot be empty.",
        )

    return {
        "category": "delivery_issue",
        "sentiment": "negative",
        "priority": "medium",
    }


@router.post("/api/v1/retrieve")
def retrieve_context_endpoint(request: TicketRequest):
    """
    Temporary endpoint.
    Replace the simulated result with Member 2's RAG function.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Ticket message cannot be empty.",
        )

    return {
        "is_relevant": True,
        "results": [
            {
                "text": (
                    "Delivery times may vary depending on "
                    "the shipping method."
                ),
                "document": "shipping_policy.pdf",
                "section": "Delivery times",
                "score": 0.85,
            }
        ],
    }


@router.post(
    "/api/v1/process-ticket",
    response_model=TicketResponse,
)
def process_ticket_endpoint(request: TicketRequest):
    """
    Temporary end-to-end endpoint.
    Real analysis, retrieval and generation will be integrated later.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Ticket message cannot be empty.",
        )

    return {
        "ticket_id": request.ticket_id,
        "category": "delivery_issue",
        "sentiment": "negative",
        "priority": "medium",
        "suggested_reply": (
            "Thank you for contacting us. Delivery times may vary "
            "depending on the shipping method. Our support team can "
            "review your order if you provide your order number."
        ),
        "sources": [
            {
                "document": "shipping_policy.pdf",
                "section": "Delivery times",
            }
        ],
        "needs_human_review": True,
    }
