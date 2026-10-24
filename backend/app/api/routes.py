import logging
from backend.app.nlp.classification import analyze_ticket
from fastapi import APIRouter, HTTPException
from backend.app.generation.generation import generate_reply
from backend.app.rag.retriever import retrieve_context
from backend.app.schemas import (
    HealthResponse,
    TicketRequest,
    TicketResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check whether the API is running."""
    return {"status": "ok"}

@router.post("/api/v1/analyze")
def analyze_ticket_endpoint(request: TicketRequest):
    """Analyze a ticket using Member 1's NLP module."""
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Ticket message cannot be empty.",
        )

    try:
        return analyze_ticket(request.message)
    except Exception as exc:
        logger.exception("Ticket classification failed")
        raise HTTPException(
            status_code=503,
            detail="Ticket classification service unavailable.",
        ) from exc
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


@router.post(
    "/api/v1/process-ticket",
    response_model=TicketResponse,
)
def process_ticket_endpoint(request: TicketRequest):
    """Run the complete NLP -> RAG -> generation pipeline."""

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Ticket message cannot be empty.",
        )

    try:
        # 1. Analyze the customer ticket
        analysis = analyze_ticket(request.message)

        # 2. Retrieve relevant policies
        context = retrieve_context(request.message, k=3)

        # 3. Generate a grounded response
        generation = generate_reply(
            ticket=request.message,
            analysis=analysis,
            context=context,
        )

        # 4. Return the complete result
        return {
            "ticket_id": request.ticket_id,
            "category": analysis["category"],
            "sentiment": analysis["sentiment"],
            "priority": analysis["priority"],
            "suggested_reply": generation["suggested_reply"],
            "sources": generation["sources"],
            "needs_human_review": generation["needs_human_review"],
        }

    except Exception as exc:
        logger.exception("Ticket processing failed")
        raise HTTPException(
            status_code=503,
            detail="Ticket processing service unavailable.",
        ) from exc
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