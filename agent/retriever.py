"""
retriever.py — Step 2.3: Semantic Search + Metadata Filtering

Two-stage retrieval:
  Stage 1: FAISS top-K (K=30) semantic search using query embedding
  Stage 2: Post-filter by metadata constraints → return top-N (N≤10) by score

Fallback: if post-filter yields < 1 result, relax constraints one at a time.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from api.models import Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "embeddings" / "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Type code mapping for test_type field
KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Assessment Exercises": "E",
    "Simulations": "S",
    "Development & 360": "D",
}


@dataclass
class SearchConstraints:
    """Structured constraints parsed from conversation context."""
    job_levels: list[str] = field(default_factory=list)
    duration_max: Optional[int] = None
    languages: list[str] = field(default_factory=list)
    test_types: list[str] = field(default_factory=list)  # Single-letter codes: K, P, A, etc.
    adaptive: Optional[bool] = None
    remote: Optional[bool] = None

    def is_empty(self) -> bool:
        """Check if no constraints are set."""
        return (
            not self.job_levels
            and self.duration_max is None
            and not self.languages
            and not self.test_types
            and self.adaptive is None
            and self.remote is None
        )

    def to_dict(self) -> dict:
        return {
            "job_levels": self.job_levels,
            "duration_max": self.duration_max,
            "languages": self.languages,
            "test_types": self.test_types,
            "adaptive": self.adaptive,
            "remote": self.remote,
        }


# ---------------------------------------------------------------------------
# Constraint relaxation order (drop least important first)
# ---------------------------------------------------------------------------
_RELAXATION_ORDER = [
    "duration_max",
    "languages",
    "adaptive",
    "job_levels",
    "test_types",  # dropped last — most important filter
]


def _matches_constraints(item: dict, constraints: SearchConstraints) -> bool:
    """Check if a catalog item satisfies all active constraints."""

    # Job level filter: item must have at least one matching job level
    if constraints.job_levels:
        item_levels = set(item.get("job_levels", []))
        if not item_levels.intersection(constraints.job_levels):
            return False

    # Duration filter: item duration must be ≤ max
    if constraints.duration_max is not None:
        item_duration = item.get("duration_minutes")
        if item_duration is not None and item_duration > constraints.duration_max:
            return False
        # Items with no duration info pass (we don't penalize missing data)

    # Language filter: item must support at least one requested language
    if constraints.languages:
        item_langs = set(item.get("languages", []))
        # Also do case-insensitive partial matching for flexibility
        matched = False
        for req_lang in constraints.languages:
            req_lower = req_lang.lower()
            for item_lang in item_langs:
                if req_lower in item_lang.lower():
                    matched = True
                    break
            if matched:
                break
        if not matched:
            return False

    # Test type filter: item must have at least one matching type code
    if constraints.test_types:
        item_codes = set(item.get("type_codes", []))
        if not item_codes.intersection(constraints.test_types):
            return False

    # Adaptive filter
    if constraints.adaptive is not None:
        if item.get("adaptive") != constraints.adaptive:
            return False

    # Remote filter
    if constraints.remote is not None:
        if item.get("remote") != constraints.remote:
            return False

    return True


def _relax_constraints(constraints: SearchConstraints) -> list[SearchConstraints]:
    """
    Generate progressively relaxed constraint sets.
    Drops one constraint at a time in priority order.
    """
    import copy

    relaxed_versions = []
    current = copy.deepcopy(constraints)

    for field_name in _RELAXATION_ORDER:
        val = getattr(current, field_name)
        # Skip if already empty/None
        is_set = (val is not None) if not isinstance(val, list) else bool(val)
        if is_set:
            relaxed = copy.deepcopy(current)
            if isinstance(val, list):
                setattr(relaxed, field_name, [])
            else:
                setattr(relaxed, field_name, None)
            relaxed_versions.append(relaxed)
            current = relaxed  # cumulative relaxation

    return relaxed_versions


class Retriever:
    """
    Semantic search + metadata filtering over the SHL product catalog.

    Usage:
        retriever = Retriever.load("embeddings/faiss_index/")
        results = retriever.search("Java developer", constraints, n=10)
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        metadata: dict[int, dict],
        model: SentenceTransformer,
    ):
        self._index = index
        self._metadata = metadata
        self._model = model

    @classmethod
    def load(cls, index_dir: str | Path | None = None) -> "Retriever":
        """Load FAISS index, metadata, and embedding model from disk."""
        index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        index_file = index_dir / "faiss_index.bin"
        metadata_file = index_dir / "metadata.pkl"

        logger.info("Loading FAISS index from %s", index_file)
        index = faiss.read_index(str(index_file))
        logger.info("  Index loaded: %d vectors", index.ntotal)

        logger.info("Loading metadata from %s", metadata_file)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        logger.info("  Metadata loaded: %d items", len(metadata))

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("  Model loaded (dim=%d)", model.get_sentence_embedding_dimension())

        return cls(index=index, metadata=metadata, model=model)

    @property
    def catalog_size(self) -> int:
        return self._index.ntotal

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return normalized vector."""
        return self._model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

    def _semantic_search(self, query: str, top_k: int = 30) -> list[tuple[float, dict]]:
        """
        Stage 1: FAISS semantic search.
        Returns list of (score, catalog_item) tuples, sorted by score desc.
        """
        q_emb = self._embed_query(query)
        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for padding
                continue
            item = self._metadata[idx]
            results.append((float(score), item))

        return results

    def _to_recommendation(self, item: dict) -> Recommendation:
        """Convert a catalog item to a Recommendation model."""
        # Pick the primary type code (first one, or "K" as fallback)
        type_codes = item.get("type_codes", [])
        primary_code = type_codes[0] if type_codes else "K"

        return Recommendation(
            name=item["name"],
            url=item["link"],
            test_type=primary_code,
        )

    def search(
        self,
        query: str,
        constraints: SearchConstraints | None = None,
        n: int = 10,
    ) -> list[Recommendation]:
        """
        Two-stage retrieval with constraint relaxation fallback.

        Args:
            query: Natural language search query
            constraints: Optional metadata filters
            n: Max results to return (capped at 10)

        Returns:
            List of Recommendation objects (1–10 items)
        """
        n = min(n, 10)  # Hard cap

        if constraints is None:
            constraints = SearchConstraints()

        # Stage 1: Semantic search — get top-30 candidates
        candidates = self._semantic_search(query, top_k=30)
        logger.info(
            "Semantic search for '%s' returned %d candidates",
            query[:50], len(candidates),
        )

        # Stage 2: Post-filter by constraints
        if not constraints.is_empty():
            filtered = [
                (score, item) for score, item in candidates
                if _matches_constraints(item, constraints)
            ]
            logger.info(
                "After constraint filtering: %d / %d candidates",
                len(filtered), len(candidates),
            )

            # Fallback: relax constraints if too few results
            if len(filtered) < 1:
                logger.info("Too few results — relaxing constraints...")
                for relaxed in _relax_constraints(constraints):
                    filtered = [
                        (score, item) for score, item in candidates
                        if _matches_constraints(item, relaxed)
                    ]
                    logger.info(
                        "  Relaxed to %s → %d results",
                        relaxed.to_dict(), len(filtered),
                    )
                    if len(filtered) >= 1:
                        break

            # If still nothing after full relaxation, use unfiltered
            if len(filtered) < 1:
                logger.warning("No results even after full relaxation — using unfiltered")
                filtered = candidates
        else:
            filtered = candidates

        # Take top-N by score
        top_n = filtered[:n]

        # Convert to Recommendation objects
        recommendations = [self._to_recommendation(item) for _, item in top_n]

        logger.info("Returning %d recommendations", len(recommendations))
        return recommendations

    def search_raw(
        self,
        query: str,
        constraints: SearchConstraints | None = None,
        n: int = 10,
    ) -> list[tuple[float, dict]]:
        """
        Like search() but returns raw (score, catalog_item) tuples.
        Useful for debugging and the agent's comparison logic.
        """
        n = min(n, 10)
        if constraints is None:
            constraints = SearchConstraints()

        candidates = self._semantic_search(query, top_k=30)

        if not constraints.is_empty():
            filtered = [
                (score, item) for score, item in candidates
                if _matches_constraints(item, constraints)
            ]
            if len(filtered) < 1:
                for relaxed in _relax_constraints(constraints):
                    filtered = [
                        (score, item) for score, item in candidates
                        if _matches_constraints(item, relaxed)
                    ]
                    if len(filtered) >= 1:
                        break
            if len(filtered) < 1:
                filtered = candidates
        else:
            filtered = candidates

        return filtered[:n]
