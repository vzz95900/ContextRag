"""Unit tests for the multi-objective optimizer module.

These tests exercise the pure scoring/selection logic using synthetic
embeddings — no API keys, no vector store, no external services needed.
"""

import math

import numpy as np
import pytest

from app.services.optimizer import (
    compute_coverage,
    compute_relevance,
    compute_support,
    cosine_similarity,
    optimize_selection,
)


# ── cosine_similarity ───────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert cosine_similarity(a, b) == 0.0

    def test_arbitrary_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine_similarity(a, b) == pytest.approx(expected)


# ── compute_relevance ──────────────────────────────────────


class TestComputeRelevance:
    def test_relevance_identical(self):
        q = np.array([1.0, 0.0])
        d = np.array([1.0, 0.0])
        assert compute_relevance(q, d) == pytest.approx(1.0)

    def test_relevance_orthogonal(self):
        q = np.array([1.0, 0.0])
        d = np.array([0.0, 1.0])
        assert compute_relevance(q, d) == pytest.approx(0.0)


# ── compute_coverage ───────────────────────────────────────


class TestComputeCoverage:
    def test_empty_set_returns_one(self):
        d = np.array([1.0, 0.0])
        assert compute_coverage(d, []) == 1.0

    def test_identical_doc_in_set(self):
        d = np.array([1.0, 0.0])
        s = [np.array([1.0, 0.0])]
        # Cov = 1 - max(sim) = 1 - 1 = 0
        assert compute_coverage(d, s) == pytest.approx(0.0)

    def test_orthogonal_doc_in_set(self):
        d = np.array([1.0, 0.0])
        s = [np.array([0.0, 1.0])]
        # Cov = 1 - 0 = 1
        assert compute_coverage(d, s) == pytest.approx(1.0)

    def test_coverage_picks_max_similarity(self):
        d = np.array([1.0, 0.0])
        s = [
            np.array([0.0, 1.0]),   # sim = 0
            np.array([0.5, 0.5]),   # sim ≈ 0.707
        ]
        max_sim = cosine_similarity(d, s[1])
        assert compute_coverage(d, s) == pytest.approx(1.0 - max_sim)


# ── compute_support ────────────────────────────────────────


class TestComputeSupport:
    def test_empty_set_returns_zero(self):
        d = np.array([1.0, 0.0])
        assert compute_support(d, []) == 0.0

    def test_single_identical_doc(self):
        d = np.array([1.0, 0.0])
        s = [np.array([1.0, 0.0])]
        # Sup = mean([1.0]) = 1.0
        assert compute_support(d, s) == pytest.approx(1.0)

    def test_mean_similarity(self):
        d = np.array([1.0, 0.0])
        s = [
            np.array([1.0, 0.0]),  # sim = 1.0
            np.array([0.0, 1.0]),  # sim = 0.0
        ]
        # Sup = (1.0 + 0.0) / 2 = 0.5
        assert compute_support(d, s) == pytest.approx(0.5)


# ── optimize_selection ─────────────────────────────────────


def _make_candidate(embedding: list, chunk_id: str = "") -> dict:
    """Helper to build a candidate dict with an embedding."""
    return {
        "chunk_id": chunk_id or f"chunk_{hash(tuple(embedding)) % 10000}",
        "text": f"text for {chunk_id}",
        "metadata": {"filename": "test.pdf", "page_num": 1},
        "score": 0.9,
        "embedding": embedding,
    }


class TestOptimizeSelection:
    def test_returns_correct_k(self):
        candidates = [
            _make_candidate([1.0, 0.0, 0.0], "c1"),
            _make_candidate([0.0, 1.0, 0.0], "c2"),
            _make_candidate([0.0, 0.0, 1.0], "c3"),
            _make_candidate([0.5, 0.5, 0.0], "c4"),
        ]
        query_emb = [1.0, 0.0, 0.0]
        result = optimize_selection(query_emb, candidates, k=2)
        assert len(result) == 2

    def test_returns_fewer_if_not_enough_candidates(self):
        candidates = [_make_candidate([1.0, 0.0], "c1")]
        result = optimize_selection([1.0, 0.0], candidates, k=5)
        assert len(result) == 1

    def test_empty_candidates(self):
        result = optimize_selection([1.0, 0.0], [], k=3)
        assert result == []

    def test_first_selected_is_most_relevant(self):
        """With empty S, coverage=1 and support=0, so score is dominated by relevance."""
        candidates = [
            _make_candidate([0.0, 1.0], "low_rel"),
            _make_candidate([1.0, 0.0], "high_rel"),
        ]
        query_emb = [1.0, 0.0]
        result = optimize_selection(query_emb, candidates, k=1, alpha=0.5, beta=0.3, gamma=0.2)
        assert result[0]["chunk_id"] == "high_rel"

    def test_diversity_encourages_different_docs(self):
        """When beta (coverage) is high, a diverse doc is preferred over a near-duplicate."""
        candidates = [
            _make_candidate([1.0, 0.0, 0.0], "relevant"),
            _make_candidate([0.99, 0.01, 0.0], "near_duplicate"),  # very similar to 'relevant'
            _make_candidate([0.7, 0.7, 0.1], "diverse"),           # moderately relevant but different
        ]
        query_emb = [1.0, 0.0, 0.0]
        # beta > alpha+gamma so coverage outweighs relevance+support for near-duplicates
        result = optimize_selection(
            query_emb, candidates, k=2, alpha=0.2, beta=0.6, gamma=0.2,
        )
        selected_ids = {r["chunk_id"] for r in result}
        # 'relevant' should be first (highest relevance + full coverage)
        assert result[0]["chunk_id"] == "relevant"
        # 'diverse' should beat 'near_duplicate' because of higher coverage
        assert "diverse" in selected_ids

    def test_output_has_score_fields(self):
        candidates = [_make_candidate([1.0, 0.0], "c1")]
        result = optimize_selection([1.0, 0.0], candidates, k=1)
        assert "opt_score" in result[0]
        assert "opt_relevance" in result[0]
        assert "opt_coverage" in result[0]
        assert "opt_support" in result[0]

    def test_embedding_removed_from_output(self):
        candidates = [_make_candidate([1.0, 0.0], "c1")]
        result = optimize_selection([1.0, 0.0], candidates, k=1)
        assert "embedding" not in result[0]
