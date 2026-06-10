"""Multi-objective optimization-based document selection.

Replaces naive top-k retrieval with a greedy algorithm that balances
relevance, coverage (diversity), and support (agreement) when selecting
the final document set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Similarity Helpers ──────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Scoring Components ──────────────────────────────────────


def compute_relevance(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Rel(q, d) = cosine_similarity(q, d)"""
    return cosine_similarity(query_emb, doc_emb)


def compute_coverage(doc_emb: np.ndarray, selected_embs: List[np.ndarray]) -> float:
    """
    Cov(d | S) = 1 - max(similarity(d, s)) for all s in S.
    If S is empty, Cov(d | S) = 1.
    """
    if not selected_embs:
        return 1.0
    max_sim = max(cosine_similarity(doc_emb, s) for s in selected_embs)
    return 1.0 - max_sim


def compute_support(doc_emb: np.ndarray, selected_embs: List[np.ndarray]) -> float:
    """
    Sup(d, S) = (1 / len(S)) * sum(similarity(d, s)) for all s in S.
    If S is empty, Sup(d, S) = 0.
    """
    if not selected_embs:
        return 0.0
    total_sim = sum(cosine_similarity(doc_emb, s) for s in selected_embs)
    return total_sim / len(selected_embs)


# ── Greedy Optimization ────────────────────────────────────


def optimize_selection(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    k: int,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Greedy multi-objective document selection.

    From the candidate pool, iteratively selects documents that maximize:
        Score(d) = alpha * Rel(q, d) + beta * Cov(d|S) + gamma * Sup(d, S)

    Args:
        query_embedding: The query vector.
        candidates: List of candidate dicts, each must contain "embedding".
        k: Maximum number of documents to select.
        alpha: Weight for relevance.
        beta: Weight for coverage (diversity).
        gamma: Weight for support (agreement).

    Returns:
        Selected list of document dicts (up to k), each augmented with
        "opt_score", "opt_relevance", "opt_coverage", and "opt_support".
    """
    if not candidates:
        return []

    k = min(k, len(candidates))
    query_emb = np.asarray(query_embedding, dtype=np.float32)

    # Pre-convert candidate embeddings to numpy arrays
    candidate_embs = [
        np.asarray(c["embedding"], dtype=np.float32) for c in candidates
    ]

    # Pre-compute relevance scores (these don't change between iterations)
    relevance_scores = [
        compute_relevance(query_emb, emb) for emb in candidate_embs
    ]

    selected: List[Dict[str, Any]] = []
    selected_embs: List[np.ndarray] = []
    remaining_indices = list(range(len(candidates)))

    logger.debug(
        "Optimizer: selecting %d from %d candidates (α=%.2f, β=%.2f, γ=%.2f)",
        k, len(candidates), alpha, beta, gamma,
    )

    while len(selected) < k and remaining_indices:
        best_score = -float("inf")
        best_idx = -1
        best_components: Optional[Dict[str, float]] = None

        for idx in remaining_indices:
            rel = relevance_scores[idx]
            cov = compute_coverage(candidate_embs[idx], selected_embs)
            sup = compute_support(candidate_embs[idx], selected_embs)

            score = alpha * rel + beta * cov + gamma * sup

            if score > best_score:
                best_score = score
                best_idx = idx
                best_components = {
                    "opt_relevance": rel,
                    "opt_coverage": cov,
                    "opt_support": sup,
                }

        if best_idx < 0:
            break

        # Add the best candidate to the selected set
        doc = candidates[best_idx].copy()
        doc["opt_score"] = best_score
        doc.update(best_components)  # type: ignore[arg-type]

        # Remove the embedding from the output (no longer needed downstream)
        doc.pop("embedding", None)

        selected.append(doc)
        selected_embs.append(candidate_embs[best_idx])
        remaining_indices.remove(best_idx)

    logger.info(
        "Optimizer: selected %d documents (scores: %.3f – %.3f)",
        len(selected),
        selected[-1]["opt_score"] if selected else 0,
        selected[0]["opt_score"] if selected else 0,
    )

    return selected
