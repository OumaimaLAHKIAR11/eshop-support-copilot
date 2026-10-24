
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    language: str = "en"


class Source(BaseModel):
    document: str
    section: str


class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    sentiment: str
    priority: str
    suggested_reply: str
    sources: list[Source]
    needs_human_review: bool


class HealthResponse(BaseModel):
    status: str

