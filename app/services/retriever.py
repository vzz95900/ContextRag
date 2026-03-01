"""Retrieval pipeline — vector search + optional BM25 hybrid + reranker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.embedder import embed_query
from app.services.vector_store import get_vector_store


def _bm25_search(query: str, corpus: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    """Sparse retrieval using BM25 over already-retrieved chunk texts."""
    if not corpus:
        return []

    from rank_bm25 import BM25Okapi

    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    for i, doc in enumerate(corpus):
        doc["bm25_score"] = float(scores[i])

    ranked = sorted(corpus, key=lambda x: x["bm25_score"], reverse=True)
    return ranked[:top_k]


def _merge_results(
    dense: List[Dict[str, Any]],
    sparse: List[Dict[str, Any]],
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """Combine dense and sparse results using reciprocal rank fusion."""
    scores: Dict[str, float] = {}
    docs_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + alpha * (1.0 / (rank + 60))
        docs_map[cid] = doc

    for rank, doc in enumerate(sparse):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1.0 / (rank + 60))
        if cid not in docs_map:
            docs_map[cid] = doc

    for cid in docs_map:
        docs_map[cid]["fusion_score"] = scores[cid]

    merged = sorted(docs_map.values(), key=lambda x: x["fusion_score"], reverse=True)
    return merged


def retrieve(
    query: str,
    top_k: int | None = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Dense vector search
    2. (Optional) BM25 hybrid fusion
    3. Return ranked candidates
    """
    top_k = top_k or settings.retrieval_top_k
    store = get_vector_store()

    # 1. Dense search
    query_emb = embed_query(query)
    dense_results = store.search(query_emb, top_k=top_k, where=filters)

    if not dense_results:
        return []

    # 2. (Optional) Hybrid with BM25
    if settings.enable_hybrid_search:
        sparse_results = _bm25_search(query, dense_results, top_k=top_k)
        results = _merge_results(dense_results, sparse_results)
    else:
        results = dense_results

    return results[:top_k]
