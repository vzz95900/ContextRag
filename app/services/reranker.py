"""Cross-encoder reranker — rescores candidate chunks against the query."""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.config import settings

# ── Singleton CrossEncoder (loaded once) ────────────────────
_cross_encoder = None


def _get_cross_encoder():
    """Return a cached CrossEncoder instance (loads only on first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int | None = None) -> List[Dict[str, Any]]:
    """
    Rerank candidates using a cross-encoder model.

    Takes the top-K candidates from retrieval and re-scores them
    with a more accurate (but slower) cross-encoder, returning top-N.
    """
    top_n = top_n or settings.rerank_top_n

    if not settings.enable_reranker or not candidates:
        return candidates[:top_n]

    model = _get_cross_encoder()

    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]
