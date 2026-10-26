import logging
from backend.app.conversation import resolve_case
from backend.app.nlp.classification import analyze_ticket
from fastapi import APIRouter, HTTPException
from backend.app.generation.generation import generate_reply, requires_direct_review, conversational_reply
from backend.app.rag.retriever import retrieve_context, expand_policy_context
from backend.app.rag.query import prepare_search_query
from backend.app.schemas import (
    HealthResponse,
    TicketRequest,
    TicketResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def customer_context(request: TicketRequest) -> str:
    """Preserve the active issue across order numbers and follow-up questions."""
    return resolve_case(request.message, [turn.model_dump() for turn in request.history]).text


def ticket_context(request: TicketRequest, analysis: dict | None = None) -> dict:
    try:
        query = prepare_search_query(customer_context(request), request.language)
    except Exception:
        logger.exception("Query preparation failed; human review required")
        return {"is_relevant": False, "results": []}
    # Search the full request and separate questions to cover compound requests.
    import re
    queries = [query] + [part.strip() for part in re.split(r"[?;\n]+", query) if len(part.strip()) > 12][:4]
    found = {}
    for part in dict.fromkeys(queries):
        for result in retrieve_context(part, k=3)["results"]:
            key = (result["document"], result["section"])
            if key not in found or found[key]["score"] < result["score"]:
                found[key] = result
    if not found and analysis:
        hints = {
            "damaged_product": "damaged defective product arrived broken support photos",
            "delivery_issue": "delivery policy parcel tracking delay order number",
            "return_request": "return policy request return unused packaging",
            "refund_request": "refund policy payment refund timing",
        }
        hint = hints.get(analysis.get("category"))
        if hint:
            for result in retrieve_context(query + "\n" + hint, k=3)["results"]:
                found[(result["document"], result["section"])] = result
    results = sorted(found.values(), key=lambda item: item["score"], reverse=True)[:8]
    results = expand_policy_context(results)
    return {"is_relevant": bool(results), "results": results}


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


@router.post("/api/v1/retrieve")
def retrieve_endpoint(request: TicketRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Ticket message cannot be empty.")
    try:
        return ticket_context(request)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable.") from exc


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
        case = resolve_case(request.message, [turn.model_dump() for turn in request.history])
        analysis = analyze_ticket(case.text)
        if case.is_followup:
            if analysis["category"] == "other":
                origin = analyze_ticket(case.issue)
                if origin["category"] != "other":
                    analysis = {**analysis, "category": origin["category"]}
            analysis = {**analysis, "intent": "support_request", "case_followup": True,
                        "case_issue": case.issue}

        # 2. Retrieve relevant policies
        context = ({"is_relevant": False, "results": []}
                   if conversational_reply(request.message, analysis, request.language) is not None
                   else ticket_context(request, analysis))

        # 3. Generate a grounded response
        generation = generate_reply(
            ticket=request.message,
            analysis=analysis,
            context=context,
            language=request.language,
            history=[turn.model_dump() for turn in request.history],
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
