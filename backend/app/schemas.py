
from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class TicketRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=6000)
    language: Literal["en", "fr"] = "en"
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


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
