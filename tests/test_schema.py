"""
test_schema.py — Unit tests for API Pydantic models (Step 1.2)

Tests:
1. ChatRequest with valid messages parses cleanly
2. ChatResponse with 0 recommendations is valid
3. ChatResponse with 10 recommendations is valid
4. ChatResponse with 11 recommendations raises ValidationError
5. Message with invalid role raises ValidationError
6. Missing required field raises ValidationError
"""

import pytest
from pydantic import ValidationError

from api.models import ChatRequest, ChatResponse, Message, Recommendation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rec(n: int = 1) -> list[dict]:
    return [
        {
            "name": f"Assessment {i}",
            "url": f"https://www.shl.com/products/product-catalog/view/assessment-{i}/",
            "test_type": "K",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

def test_message_valid_user():
    m = Message(role="user", content="I need an assessment")
    assert m.role == "user"
    assert m.content == "I need an assessment"


def test_message_valid_assistant():
    m = Message(role="assistant", content="What role are you hiring for?")
    assert m.role == "assistant"


def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="system", content="Oops")


# ---------------------------------------------------------------------------
# ChatRequest tests
# ---------------------------------------------------------------------------

def test_chat_request_valid():
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "I need a Java developer assessment"},
        ]
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_chat_request_empty_messages():
    # Empty message list is technically allowed by schema
    req = ChatRequest(messages=[])
    assert req.messages == []


def test_chat_request_missing_messages():
    with pytest.raises(ValidationError):
        ChatRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------

def test_recommendation_valid():
    rec = Recommendation(
        name="OPQ32r",
        url="https://www.shl.com/products/product-catalog/view/opq32r/",
        test_type="P",
    )
    assert rec.test_type == "P"


# ---------------------------------------------------------------------------
# ChatResponse tests
# ---------------------------------------------------------------------------

def test_chat_response_no_recommendations():
    resp = ChatResponse(
        reply="What role are you hiring for?",
        recommendations=[],
        end_of_conversation=False,
    )
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_chat_response_ten_recommendations():
    resp = ChatResponse(
        reply="Here are 10 assessments.",
        recommendations=[Recommendation(**r) for r in _make_rec(10)],
        end_of_conversation=True,
    )
    assert len(resp.recommendations) == 10


def test_chat_response_eleven_recommendations_raises():
    """Schema must reject more than 10 recommendations."""
    with pytest.raises(ValidationError) as exc_info:
        ChatResponse(
            reply="Too many",
            recommendations=[Recommendation(**r) for r in _make_rec(11)],
            end_of_conversation=False,
        )
    assert "10" in str(exc_info.value)


def test_chat_response_missing_reply_raises():
    with pytest.raises(ValidationError):
        ChatResponse(  # type: ignore[call-arg]
            recommendations=[],
            end_of_conversation=False,
        )


def test_chat_response_serializes_to_dict():
    """Ensure the response can round-trip through JSON."""
    resp = ChatResponse(
        reply="Hello",
        recommendations=[
            Recommendation(
                name="Verify G+",
                url="https://www.shl.com/products/product-catalog/view/verify-g-plus/",
                test_type="A",
            )
        ],
        end_of_conversation=False,
    )
    d = resp.model_dump()
    assert d["recommendations"][0]["name"] == "Verify G+"
    assert d["end_of_conversation"] is False
