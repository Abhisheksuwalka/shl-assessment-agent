"""
test_retriever.py — Unit tests for the Retriever (Step 2.3)

Tests:
1. Java developer query returns Java assessments
2. Personality + Executive filter returns OPQ-family assessments
3. Duration filter (max 10 min) excludes long assessments
4. Language filter (Spanish) returns only Spanish-capable assessments
5. Empty constraints returns results without error
6. Constraint relaxation works when filters are too restrictive
7. Results are capped at 10
8. search_raw returns tuples with scores
"""

import pytest
from agent.retriever import Retriever, SearchConstraints, _matches_constraints


# ---------------------------------------------------------------------------
# Fixtures — load retriever once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    """Load the retriever once for the entire test module."""
    return Retriever.load()


# ---------------------------------------------------------------------------
# SearchConstraints tests
# ---------------------------------------------------------------------------

class TestSearchConstraints:

    def test_empty_constraints(self):
        c = SearchConstraints()
        assert c.is_empty()

    def test_non_empty_with_job_levels(self):
        c = SearchConstraints(job_levels=["Mid-Professional"])
        assert not c.is_empty()

    def test_non_empty_with_duration(self):
        c = SearchConstraints(duration_max=20)
        assert not c.is_empty()

    def test_to_dict(self):
        c = SearchConstraints(job_levels=["Executive"], test_types=["P"])
        d = c.to_dict()
        assert d["job_levels"] == ["Executive"]
        assert d["test_types"] == ["P"]
        assert d["duration_max"] is None


# ---------------------------------------------------------------------------
# _matches_constraints unit tests
# ---------------------------------------------------------------------------

class TestMatchesConstraints:

    def test_matches_job_level(self):
        item = {"job_levels": ["Mid-Professional", "Manager"], "type_codes": ["K"]}
        c = SearchConstraints(job_levels=["Mid-Professional"])
        assert _matches_constraints(item, c)

    def test_no_match_job_level(self):
        item = {"job_levels": ["Entry-Level"], "type_codes": ["K"]}
        c = SearchConstraints(job_levels=["Executive"])
        assert not _matches_constraints(item, c)

    def test_matches_duration(self):
        item = {"duration_minutes": 15, "type_codes": ["K"]}
        c = SearchConstraints(duration_max=20)
        assert _matches_constraints(item, c)

    def test_no_match_duration(self):
        item = {"duration_minutes": 30, "type_codes": ["K"]}
        c = SearchConstraints(duration_max=20)
        assert not _matches_constraints(item, c)

    def test_none_duration_passes(self):
        """Items without duration info should pass the filter."""
        item = {"duration_minutes": None, "type_codes": ["K"]}
        c = SearchConstraints(duration_max=20)
        assert _matches_constraints(item, c)

    def test_matches_language_exact(self):
        item = {"languages": ["English (USA)", "French"], "type_codes": ["K"]}
        c = SearchConstraints(languages=["French"])
        assert _matches_constraints(item, c)

    def test_matches_language_partial(self):
        """Partial case-insensitive match: 'spanish' matches 'Latin American Spanish'."""
        item = {"languages": ["Latin American Spanish"], "type_codes": ["K"]}
        c = SearchConstraints(languages=["Spanish"])
        assert _matches_constraints(item, c)

    def test_no_match_language(self):
        item = {"languages": ["English (USA)"], "type_codes": ["K"]}
        c = SearchConstraints(languages=["Japanese"])
        assert not _matches_constraints(item, c)

    def test_matches_test_type(self):
        item = {"type_codes": ["K", "P"]}
        c = SearchConstraints(test_types=["P"])
        assert _matches_constraints(item, c)

    def test_no_match_test_type(self):
        item = {"type_codes": ["K"]}
        c = SearchConstraints(test_types=["P"])
        assert not _matches_constraints(item, c)

    def test_matches_adaptive(self):
        item = {"adaptive": True, "type_codes": ["K"]}
        c = SearchConstraints(adaptive=True)
        assert _matches_constraints(item, c)

    def test_no_match_adaptive(self):
        item = {"adaptive": False, "type_codes": ["K"]}
        c = SearchConstraints(adaptive=True)
        assert not _matches_constraints(item, c)

    def test_empty_constraints_always_match(self):
        item = {"type_codes": ["K"]}
        c = SearchConstraints()
        assert _matches_constraints(item, c)


# ---------------------------------------------------------------------------
# Retriever.search() integration tests
# ---------------------------------------------------------------------------

class TestRetrieverSearch:

    def test_java_developer(self, retriever):
        """Java developer query should return Java-related assessments."""
        results = retriever.search("Java developer assessment")
        assert len(results) > 0
        assert len(results) <= 10
        names = [r.name.lower() for r in results]
        assert any("java" in name for name in names), (
            f"Expected 'java' in results, got: {names}"
        )

    def test_personality_executive(self, retriever):
        """Personality + Executive filter should return personality assessments."""
        constraints = SearchConstraints(
            job_levels=["Executive"],
            test_types=["P"],
        )
        results = retriever.search(
            "personality test for executives",
            constraints=constraints,
        )
        assert len(results) > 0
        # All results should have test_type P
        for r in results:
            assert r.test_type == "P", f"Expected test_type 'P', got '{r.test_type}' for {r.name}"

    def test_duration_filter(self, retriever):
        """Duration filter (max 10 min) should not include long assessments."""
        constraints = SearchConstraints(duration_max=10)
        results = retriever.search_raw(
            "assessment test",
            constraints=constraints,
        )
        assert len(results) > 0
        for _, item in results:
            dur = item.get("duration_minutes")
            if dur is not None:
                assert dur <= 10, f"Duration {dur} exceeds max 10 for {item['name']}"

    def test_spanish_language_filter(self, retriever):
        """Spanish language filter should only return Spanish-capable assessments."""
        constraints = SearchConstraints(languages=["Spanish"])
        results = retriever.search_raw(
            "technical skills assessment",
            constraints=constraints,
        )
        assert len(results) > 0
        for _, item in results:
            langs = [l.lower() for l in item.get("languages", [])]
            assert any("spanish" in l for l in langs), (
                f"Expected Spanish in languages for {item['name']}, got: {item.get('languages', [])}"
            )

    def test_empty_constraints(self, retriever):
        """Search with no constraints should return results."""
        results = retriever.search("cognitive ability")
        assert len(results) > 0
        assert len(results) <= 10

    def test_results_capped_at_ten(self, retriever):
        """Results must never exceed 10."""
        results = retriever.search("assessment", n=20)
        assert len(results) <= 10

    def test_constraint_relaxation(self, retriever):
        """Very restrictive constraints should still return results via relaxation."""
        constraints = SearchConstraints(
            job_levels=["Executive"],
            test_types=["E"],  # Very rare (only 2 items)
            languages=["Japanese"],  # Very restrictive
            duration_max=5,
        )
        results = retriever.search(
            "assessment exercise",
            constraints=constraints,
        )
        # Should still return something via relaxation
        assert len(results) > 0

    def test_search_raw_returns_tuples(self, retriever):
        """search_raw should return (score, item) tuples."""
        results = retriever.search_raw("Python developer")
        assert len(results) > 0
        score, item = results[0]
        assert isinstance(score, float)
        assert isinstance(item, dict)
        assert "name" in item
        assert "link" in item

    def test_recommendation_has_valid_url(self, retriever):
        """All recommendations must have real catalog URLs."""
        results = retriever.search("data analysis")
        for r in results:
            assert r.url.startswith("https://www.shl.com/"), (
                f"URL does not look like SHL catalog: {r.url}"
            )
            assert r.name  # Non-empty name
            assert r.test_type in ("K", "P", "A", "B", "C", "E", "S", "D"), (
                f"Unknown test_type '{r.test_type}' for {r.name}"
            )

    def test_mid_professional_java(self, retriever):
        """Java dev, mid-professional should return Java assessments at right level."""
        constraints = SearchConstraints(
            job_levels=["Mid-Professional"],
            test_types=["K"],
        )
        results = retriever.search(
            "Java developer mid-level",
            constraints=constraints,
        )
        assert len(results) > 0
        names = [r.name.lower() for r in results]
        assert any("java" in name for name in names)
