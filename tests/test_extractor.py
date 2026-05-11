"""
test_extractor.py — Unit tests for the Context Extractor (Step 3.2)

Uses mocked Groq responses to test extractor logic without real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock

from api.models import Message
from agent.extractor import (
    extract_constraints,
    build_search_query,
    _extract_json,
    _parse_constraints,
    _merge_constraints,
)
from agent.retriever import SearchConstraints


def _mock_groq_response(content: str):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


def _make_mock_client(json_output: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_groq_response(
        json.dumps(json_output)
    )
    return client


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_direct_json(self):
        assert _extract_json('{"job_role": "SE"}')["job_role"] == "SE"

    def test_markdown_fenced_json(self):
        assert _extract_json('```json\n{"job_role": "DA"}\n```')["job_role"] == "DA"

    def test_bare_json_in_text(self):
        assert _extract_json('Result: {"job_role": "M"}')["job_role"] == "M"

    def test_invalid_json_returns_empty(self):
        assert _extract_json("not json") == {}

    def test_empty_string_returns_empty(self):
        assert _extract_json("") == {}


# ---------------------------------------------------------------------------
# _parse_constraints tests
# ---------------------------------------------------------------------------

class TestParseConstraints:
    def test_valid_job_levels(self):
        c = _parse_constraints({"job_levels": ["Mid-Professional", "Manager"]})
        assert set(c.job_levels) == {"Mid-Professional", "Manager"}

    def test_invalid_job_levels_filtered(self):
        c = _parse_constraints({"job_levels": ["Mid-Professional", "INVALID"]})
        assert c.job_levels == ["Mid-Professional"]

    def test_valid_test_types(self):
        c = _parse_constraints({"test_types": ["K", "P"]})
        assert set(c.test_types) == {"K", "P"}

    def test_invalid_test_types_filtered(self):
        c = _parse_constraints({"test_types": ["K", "X"]})
        assert c.test_types == ["K"]

    def test_duration_positive_int(self):
        assert _parse_constraints({"duration_max": 20}).duration_max == 20

    def test_duration_negative_is_none(self):
        assert _parse_constraints({"duration_max": -5}).duration_max is None

    def test_duration_string_parsed(self):
        assert _parse_constraints({"duration_max": "30"}).duration_max == 30

    def test_duration_invalid_string_is_none(self):
        assert _parse_constraints({"duration_max": "abc"}).duration_max is None

    def test_languages_passed_through(self):
        c = _parse_constraints({"languages": ["French", "Spanish"]})
        assert set(c.languages) == {"French", "Spanish"}

    def test_adaptive_bool(self):
        assert _parse_constraints({"adaptive": True}).adaptive is True

    def test_adaptive_non_bool_is_none(self):
        assert _parse_constraints({"adaptive": "yes"}).adaptive is None

    def test_empty_data_returns_empty(self):
        assert _parse_constraints({}).is_empty()

    def test_null_values_handled(self):
        c = _parse_constraints({"job_levels": None, "test_types": None, "duration_max": None})
        assert c.is_empty()


# ---------------------------------------------------------------------------
# _merge_constraints tests
# ---------------------------------------------------------------------------

class TestMergeConstraints:
    def test_lists_are_unioned(self):
        base = SearchConstraints(job_levels=["Manager"], test_types=["K"])
        new = SearchConstraints(job_levels=["Executive"], test_types=["P"])
        merged = _merge_constraints(base, new)
        assert set(merged.job_levels) == {"Manager", "Executive"}
        assert set(merged.test_types) == {"K", "P"}

    def test_scalar_prefers_new(self):
        merged = _merge_constraints(
            SearchConstraints(duration_max=20),
            SearchConstraints(duration_max=30),
        )
        assert merged.duration_max == 30

    def test_scalar_keeps_base_if_new_is_none(self):
        merged = _merge_constraints(
            SearchConstraints(duration_max=20, adaptive=True),
            SearchConstraints(),
        )
        assert merged.duration_max == 20
        assert merged.adaptive is True

    def test_languages_unioned(self):
        merged = _merge_constraints(
            SearchConstraints(languages=["English (USA)"]),
            SearchConstraints(languages=["French"]),
        )
        assert set(merged.languages) == {"English (USA)", "French"}


# ---------------------------------------------------------------------------
# extract_constraints integration (mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractConstraints:
    def test_single_turn(self):
        msgs = [Message(role="user", content="Java dev, mid-level")]
        mock_out = {"job_role": "SE", "skills": ["Java"], "job_levels": ["Mid-Professional"],
                    "test_types": ["K"], "duration_max": None, "languages": [], "adaptive": None}
        constraints, summary = extract_constraints(msgs, client=_make_mock_client(mock_out))
        assert "Mid-Professional" in constraints.job_levels
        assert "K" in constraints.test_types

    def test_refine_merge(self):
        msgs = [
            Message(role="user", content="Python developer assessments"),
            Message(role="assistant", content="Here are some..."),
            Message(role="user", content="Also add personality tests"),
        ]
        previous = SearchConstraints(test_types=["K"], job_levels=["Mid-Professional"])
        mock_out = {"job_role": "SE", "skills": ["Python"], "job_levels": ["Mid-Professional"],
                    "test_types": ["P"], "duration_max": None, "languages": [], "adaptive": None}
        constraints, _ = extract_constraints(msgs, previous_constraints=previous,
                                             client=_make_mock_client(mock_out))
        assert set(constraints.test_types) == {"K", "P"}

    def test_llm_failure_returns_empty(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        constraints, summary = extract_constraints(
            [Message(role="user", content="test")], client=client
        )
        assert constraints.is_empty()
        assert "failed" in summary.lower()

    def test_spanish_language(self):
        msgs = [Message(role="user", content="Spanish-speaking candidates")]
        mock_out = {"languages": ["Latin American Spanish"], "job_levels": [],
                    "test_types": [], "duration_max": None, "adaptive": None}
        constraints, _ = extract_constraints(msgs, client=_make_mock_client(mock_out))
        assert "Latin American Spanish" in constraints.languages


# ---------------------------------------------------------------------------
# build_search_query tests
# ---------------------------------------------------------------------------

class TestBuildSearchQuery:
    def test_basic(self):
        q = build_search_query([Message(role="user", content="Java dev")], "Role: SE; Types: K")
        assert "Java dev" in q and "Role: SE" in q

    def test_empty_constraints(self):
        q = build_search_query([Message(role="user", content="Help")],
                               "No specific constraints extracted.")
        assert q == "Help"

    def test_multi_turn_uses_last(self):
        msgs = [Message(role="user", content="Help"),
                Message(role="assistant", content="What role?"),
                Message(role="user", content="Java developer")]
        q = build_search_query(msgs, "Role: Java")
        assert "Java developer" in q
