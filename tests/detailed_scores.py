"""
Detailed per-chunk scoring table for research paper.

Shows exactly which chunks are selected and their Rel/Cov/Sup at each
greedy iteration step — for both Top-K and Optimized pipelines.

Usage:
    python tests/detailed_scores.py                    # first query
    python tests/detailed_scores.py --query-index 2    # specific query
    python tests/detailed_scores.py --all               # all queries
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.embedder import embed_query
from app.services.optimizer import (
    compute_coverage,
    compute_relevance,
    compute_support,
    cosine_similarity,
)
from app.services.vector_store import get_vector_store

QUERIES_PATH = Path(__file__).parent / "eval_queries.json"


def compute_per_doc_scores(query_emb, doc_embs):
    """Simulate greedy selection and return per-step Rel/Cov/Sup."""
    rows = []
    built = []
    for step, emb in enumerate(doc_embs):
        rel = compute_relevance(query_emb, emb)
        cov = compute_coverage(emb, built)
        sup = compute_support(emb, built)
        score = settings.optimizer_alpha * rel + settings.optimizer_beta * cov + settings.optimizer_gamma * sup
        rows.append({"step": step + 1, "rel": rel, "cov": cov, "sup": sup, "score": score})
        built.append(emb)
    return rows


def run_query(query_text, query_idx, top_k=10, candidate_n=30):
    store = get_vector_store()
    query_emb = embed_query(query_text)
    q = np.asarray(query_emb, dtype=np.float32)

    # Top-K: just take the top-k by cosine similarity
    topk_results = store.search_with_embeddings(query_emb, top_k=top_k)
    topk_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in topk_results]
    topk_ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(topk_results)]

    # Optimized: larger pool → greedy selection
    candidates = store.search_with_embeddings(query_emb, top_k=candidate_n)
    cand_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in candidates]

    # Run greedy optimizer manually to track which chunks are selected
    alpha, beta, gamma = settings.optimizer_alpha, settings.optimizer_beta, settings.optimizer_gamma
    remaining = list(range(len(candidates)))
    selected_indices = []
    selected_embs = []

    for _ in range(min(top_k, len(candidates))):
        best_score, best_idx = -float("inf"), -1
        for idx in remaining:
            rel = compute_relevance(q, cand_embs[idx])
            cov = compute_coverage(cand_embs[idx], selected_embs)
            sup = compute_support(cand_embs[idx], selected_embs)
            score = alpha * rel + beta * cov + gamma * sup
            if score > best_score:
                best_score = score
                best_idx = idx
        selected_indices.append(best_idx)
        selected_embs.append(cand_embs[best_idx])
        remaining.remove(best_idx)

    opt_ids = [candidates[i].get("chunk_id", f"cand_{i}") for i in selected_indices]
    opt_embs_ordered = [cand_embs[i] for i in selected_indices]

    # Compute per-doc scores
    topk_scores = compute_per_doc_scores(q, topk_embs)
    opt_scores = compute_per_doc_scores(q, opt_embs_ordered)

    # Print
    short_q = query_text[:75]
    print(f"\n{'=' * 110}")
    print(f"  Q{query_idx}: {short_q}")
    print(f"{'=' * 110}")

    print(f"\n  {'─' * 106}")
    print(f"  STANDARD TOP-K SELECTION (k={top_k})")
    print(f"  {'─' * 106}")
    print(f"  {'Step':>4}   {'Chunk ID':<30}   {'Rel(q,d)':>10}   {'Cov(d|S)':>10}   {'Sup(d,S)':>10}   {'Score':>10}")
    print(f"  {'─' * 106}")

    for row, cid in zip(topk_scores, topk_ids):
        short_id = cid[:28]
        print(f"  {row['step']:>4}   {short_id:<30}   {row['rel']:>10.4f}   {row['cov']:>10.4f}   {row['sup']:>10.4f}   {row['score']:>10.4f}")

    avg_r = np.mean([r["rel"] for r in topk_scores])
    avg_c = np.mean([r["cov"] for r in topk_scores])
    avg_s = np.mean([r["sup"] for r in topk_scores])
    avg_sc = np.mean([r["score"] for r in topk_scores])
    print(f"  {'─' * 106}")
    print(f"  {'AVG':>4}   {'':<30}   {avg_r:>10.4f}   {avg_c:>10.4f}   {avg_s:>10.4f}   {avg_sc:>10.4f}")

    print(f"\n  {'─' * 106}")
    print(f"  MULTI-OBJECTIVE OPTIMIZED SELECTION (k={top_k}, N={candidate_n})")
    print(f"  {'─' * 106}")
    print(f"  {'Step':>4}   {'Chunk ID':<30}   {'Rel(q,d)':>10}   {'Cov(d|S)':>10}   {'Sup(d,S)':>10}   {'Score':>10}")
    print(f"  {'─' * 106}")

    for row, cid in zip(opt_scores, opt_ids):
        short_id = cid[:28]
        print(f"  {row['step']:>4}   {short_id:<30}   {row['rel']:>10.4f}   {row['cov']:>10.4f}   {row['sup']:>10.4f}   {row['score']:>10.4f}")

    avg_r2 = np.mean([r["rel"] for r in opt_scores])
    avg_c2 = np.mean([r["cov"] for r in opt_scores])
    avg_s2 = np.mean([r["sup"] for r in opt_scores])
    avg_sc2 = np.mean([r["score"] for r in opt_scores])
    print(f"  {'─' * 106}")
    print(f"  {'AVG':>4}   {'':<30}   {avg_r2:>10.4f}   {avg_c2:>10.4f}   {avg_s2:>10.4f}   {avg_sc2:>10.4f}")
    print(f"  {'=' * 110}")


def main():
    parser = argparse.ArgumentParser(description="Detailed per-chunk scoring for research paper")
    parser.add_argument("--query-index", type=int, default=0, help="Which query to show (0-indexed)")
    parser.add_argument("--all", action="store_true", help="Show all queries")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if args.all:
        for i, q in enumerate(queries):
            run_query(q["query"], i + 1, top_k=args.top_k)
    else:
        idx = args.query_index
        if idx >= len(queries):
            print(f"Only {len(queries)} queries available (0-indexed)")
            sys.exit(1)
        run_query(queries[idx]["query"], idx + 1, top_k=args.top_k)


if __name__ == "__main__":
    main()
