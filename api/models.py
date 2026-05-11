"""
Pydantic models for the SHL Assessment Recommendation API.
This is the non-negotiable schema evaluated by the automated harness.
"""

from typing import List, Literal
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """A single message in a conversation turn."""
    role: Literal["user", "assistant"]
    content: str


class Recommendation(BaseModel):
    """
    A single SHL assessment recommendation.
    URLs must be verbatim from the SHL catalog — never hallucinated.
    """
    name: str
    url: str
    test_type: str  # Single-letter code: K, P, A, B, C, E, S, D


class ChatRequest(BaseModel):
    """Full stateless conversation history sent on every POST /chat request."""
    messages: List[Message]


class ChatResponse(BaseModel):
    """
    The response schema returned by POST /chat.
    This schema is non-negotiable and validated by the automated harness.
    """
    reply: str
    recommendations: List[Recommendation]  # empty [] or 1–10 items
    end_of_conversation: bool

    @field_validator("recommendations")
    @classmethod
    def recommendations_max_ten(cls, v: List[Recommendation]) -> List[Recommendation]:
        if len(v) > 10:
            raise ValueError(
                f"recommendations must contain at most 10 items, got {len(v)}"
            )
        return v
