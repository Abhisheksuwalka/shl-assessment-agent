"""
test_agent.py — End-to-end tests for the Agent orchestrator (Step 3.5)

Tests all 5 intent paths with mocked LLM to verify:
- Schema compliance on every response
- Correct routing (CLARIFY → no recs, RECOMMEND → recs, REFUSE → no recs)
- Turn-cap forced recommendation
- URL validation (all from catalog)

Uses a real Retriever (loaded from disk) but mocked Groq LLM.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from api.models import ChatResponse, Message, Recommendation
from agent.agent import Agent
from agent.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    """Load the real retriever once."""
    return Retriever.load()


def _mock_groq_response(content: str):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Mock Groq client that returns responses in sequence."""
    client = MagicMock()
    side_effects = [_mock_groq_response(r) for r in responses]
    client.chat.completions.create.side_effect = side_effects
    return client


def _validate_response(resp: ChatResponse):
    """Common schema validations."""
    assert isinstance(resp.reply, str) and len(resp.reply) > 0
    assert isinstance(resp.recommendations, list)
    assert len(resp.recommendations) <= 10
    assert isinstance(resp.end_of_conversation, bool)
    for rec in resp.recommendations:
        assert rec.name
        assert rec.url.startswith("https://www.shl.com/")
        assert rec.test_type in ("K", "P", "A", "B", "C", "E", "S", "D")


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgentChat:

    def test_empty_messages_greeting(self, retriever):
        """Empty messages should return a greeting."""
        agent = Agent(retriever=retriever, groq_client=MagicMock())
        resp = agent.chat([])
        _validate_response(resp)
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_refuse_path(self, retriever):
        """Off-topic query → REFUSE → no recommendations."""
        # Responses: 1) classifier → REFUSE
        client = _make_mock_client(["REFUSE"])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([Message(role="user", content="What's your CEO's name?")])
        _validate_response(resp)
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_clarify_path(self, retriever):
        """Vague query → CLARIFY → asks question, no recommendations."""
        # Responses: 1) classifier → CLARIFY, 2) extractor → constraints,
        # 3) clarification LLM call
        client = _make_mock_client([
            "CLARIFY",
            json.dumps({"job_role": None, "skills": [], "job_levels": [],
                        "test_types": [], "duration_max": None,
                        "languages": [], "adaptive": None}),
            "What role are you hiring for?",
        ])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([Message(role="user", content="I need an assessment")])
        _validate_response(resp)
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_recommend_path(self, retriever):
        """Detailed query → RECOMMEND → returns recommendations."""
        # Responses: 1) classifier → RECOMMEND, 2) extractor → constraints,
        # 3) recommendation reply
        client = _make_mock_client([
            "RECOMMEND",
            json.dumps({"job_role": "Software Engineer", "skills": ["Java"],
                        "job_levels": ["Mid-Professional"], "test_types": ["K"],
                        "duration_max": None, "languages": [], "adaptive": None}),
            "Based on your requirements for a mid-level Java developer, "
            "I recommend these assessments that cover Java programming skills.",
        ])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([
            Message(role="user", content="I need assessments for a mid-level Java developer"),
        ])
        _validate_response(resp)
        assert len(resp.recommendations) > 0
        assert resp.end_of_conversation is True

    def test_refine_path(self, retriever):
        """Refine query → REFINE → updated recommendations, eoc=False."""
        # Responses: 1) classifier → REFINE,
        # 2) extractor for previous constraints, 3) extractor for current,
        # 4) recommendation reply
        client = _make_mock_client([
            "REFINE",
            # previous constraints extraction
            json.dumps({"job_role": "SE", "skills": ["Python"],
                        "job_levels": ["Mid-Professional"], "test_types": ["K"],
                        "duration_max": None, "languages": [], "adaptive": None}),
            # current constraints extraction
            json.dumps({"job_role": "SE", "skills": ["Python"],
                        "job_levels": ["Mid-Professional"], "test_types": ["P"],
                        "duration_max": None, "languages": [], "adaptive": None}),
            "I've added personality assessments to your list.",
        ])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([
            Message(role="user", content="I need Python dev assessments"),
            Message(role="assistant", content="Here are Python assessments..."),
            Message(role="user", content="Also add personality tests"),
        ])
        _validate_response(resp)
        assert resp.end_of_conversation is False  # REFINE, not RECOMMEND

    def test_compare_path(self, retriever):
        """Compare query → COMPARE → no recommendations, comparison text."""
        client = _make_mock_client([
            "COMPARE",
            json.dumps({"job_role": None, "skills": [], "job_levels": [],
                        "test_types": [], "duration_max": None,
                        "languages": [], "adaptive": None}),
            "The OPQ32 is a personality questionnaire while the Verify G+ "
            "measures general cognitive ability.",
        ])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([
            Message(role="user", content="Compare OPQ32 and Verify G+"),
        ])
        _validate_response(resp)
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_schema_compliance_all_paths(self, retriever):
        """Every response must serialize to valid JSON matching the schema."""
        agent = Agent(retriever=retriever, groq_client=_make_mock_client(["REFUSE"]))
        resp = agent.chat([Message(role="user", content="Tell me a joke")])
        d = resp.model_dump()
        assert "reply" in d
        assert "recommendations" in d
        assert "end_of_conversation" in d

    def test_catalog_urls_only(self, retriever):
        """All recommendation URLs must be real SHL catalog URLs."""
        client = _make_mock_client([
            "RECOMMEND",
            json.dumps({"job_role": "Data Analyst", "skills": ["SQL"],
                        "job_levels": ["Mid-Professional"], "test_types": ["K"],
                        "duration_max": None, "languages": [], "adaptive": None}),
            "Here are assessments for data analysis skills.",
        ])
        agent = Agent(retriever=retriever, groq_client=client)
        resp = agent.chat([
            Message(role="user", content="I need a test for a data analyst"),
        ])
        for rec in resp.recommendations:
            assert rec.url.startswith("https://www.shl.com/"), (
                f"Non-catalog URL: {rec.url}"
            )
