"""
Evaluation script — measures real Impact on Answer Quality metrics.

Runs both Top-K and Optimized retrieval pipelines against the live
ChromaDB and computes:
  1. Evidence Diversity    (embedding-based)
  2. Redundancy Reduction  (embedding-based)
  3. Hallucination Resistance (LLM-as-judge)
  4. Answer Reliability      (LLM-as-judge)

Usage:
    cd d:\\Pjts\\CntextAware
    python tests/evaluate_quality.py              # full run
    python tests/evaluate_quality.py --dry-run    # verify script loads OK
    python tests/evaluate_quality.py --skip-llm   # skip LLM-based metrics (faster, free)
    python tests/evaluate_quality.py --scores-only # only Rel/Cov/Sup table (fastest, free)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.embedder import embed_query
from app.services.optimizer import (
    compute_coverage,
    compute_relevance,
    compute_support,
    cosine_similarity,
    optimize_selection,
)
from app.services.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# Threshold for considering two chunks "near-duplicates"
REDUNDANCY_THRESHOLD = 0.85


# ── Embedding-Based Metrics ────────────────────────────────


def _get_embeddings(chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Extract embedding vectors from chunks (must have 'embedding' key)."""
    return [np.asarray(c["embedding"], dtype=np.float32) for c in chunks]


def measure_diversity(embeddings: List[np.ndarray]) -> float:
    """
    Evidence Diversity = (1 - mean pairwise cosine similarity) × 100.

    Higher = more diverse selected set.
    """
    if len(embeddings) < 2:
        return 100.0

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))

    return (1.0 - float(np.mean(sims))) * 100


def count_near_duplicates(embeddings: List[np.ndarray], threshold: float = REDUNDANCY_THRESHOLD) -> int:
    """Count pairs with similarity above threshold."""
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                count += 1
    return count


def measure_redundancy_reduction(topk_embs: List[np.ndarray], opt_embs: List[np.ndarray]) -> float:
    """
    Redundancy Reduction = (topk_dupes - opt_dupes) / topk_dupes × 100.

    How much the optimizer reduces near-duplicate pairs compared to top-k.
    """
    topk_dupes = count_near_duplicates(topk_embs)
    opt_dupes = count_near_duplicates(opt_embs)

    if topk_dupes == 0:
        # Top-k had no duplicates — can't measure reduction
        return 100.0 if opt_dupes == 0 else 0.0

    return ((topk_dupes - opt_dupes) / topk_dupes) * 100


def compute_scoring_components(
    query_emb: np.ndarray,
    selected_embs: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute average Relevance, Coverage, and Support for a selected set.

    Simulates the greedy selection process to compute what each doc's
    Cov and Sup would have been at the time it was selected.
    """
    if not selected_embs:
        return {"avg_rel": 0.0, "avg_cov": 0.0, "avg_sup": 0.0}

    rels, covs, sups = [], [], []
    built_set: List[np.ndarray] = []

    for emb in selected_embs:
        rel = compute_relevance(query_emb, emb)
        cov = compute_coverage(emb, built_set)
        sup = compute_support(emb, built_set)

        rels.append(rel)
        covs.append(cov)
        sups.append(sup)
        built_set.append(emb)

    return {
        "avg_rel": round(float(np.mean(rels)), 4),
        "avg_cov": round(float(np.mean(covs)), 4),
        "avg_sup": round(float(np.mean(sups)), 4),
    }


# ── LLM-as-Judge Metrics ──────────────────────────────────


FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given a CONTEXT (retrieved document chunks) and an ANSWER generated from that context, score the answer's faithfulness.

Faithfulness = what percentage of claims in the answer are directly supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0-100>, "reason": "<one sentence>"}}
"""

RELIABILITY_PROMPT = """\
You are an evaluation judge. Given a QUESTION and an ANSWER, score the answer's completeness and reliability.

Consider:
- Does it address all aspects of the question?
- Is it structured and actionable?
- Does it avoid vague or generic statements?

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0-100>, "reason": "<one sentence>"}}
"""


def _call_llm_judge(prompt: str) -> Dict[str, Any]:
    """Call the configured LLM provider with a judge prompt and parse JSON response."""
    from app.services.llm_chain import generate_answer

    # Use generate_answer with a dummy chunk containing the prompt
    dummy_chunks = [{"text": "Evaluation task.", "metadata": {"filename": "eval", "page_num": 0}}]

    try:
        raw = generate_answer(query=prompt, chunks=dummy_chunks, history=None)
        # Try to parse JSON from the response
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        result = json.loads(cleaned)
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("LLM judge parse error: %s — raw: %.200s", e, raw if 'raw' in dir() else "N/A")
        return {"score": 50, "reason": f"Parse error: {e}"}


def measure_faithfulness(context_text: str, answer: str) -> float:
    """Hallucination Resistance via LLM-as-judge faithfulness scoring."""
    prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    result = _call_llm_judge(prompt)
    return float(result.get("score", 50))


def measure_reliability(question: str, answer: str) -> float:
    """Answer Reliability via LLM-as-judge completeness scoring."""
    prompt = RELIABILITY_PROMPT.format(question=question, answer=answer)
    result = _call_llm_judge(prompt)
    return float(result.get("score", 50))


# ── Pipeline Runners ───────────────────────────────────────


def run_topk_pipeline(query: str, top_k: int = 10, skip_answer: bool = False) -> Tuple[List[Dict], str]:
    """Run the standard top-k retrieval pipeline. Returns (chunks_with_embeddings, answer)."""
    store = get_vector_store()
    query_emb = embed_query(query)

    # Get results WITH embeddings (so we can compute metrics)
    results = store.search_with_embeddings(query_emb, top_k=top_k)

    if skip_answer:
        return results, ""

    # Generate answer from the top-k chunks
    from app.services.llm_chain import generate_answer

    # Make copies without embeddings for answer generation
    chunks_for_llm = []
    for r in results:
        c = r.copy()
        c.pop("embedding", None)
        chunks_for_llm.append(c)

    answer = generate_answer(query=query, chunks=chunks_for_llm)
    return results, answer


def run_optimized_pipeline(query: str, top_k: int = 10, candidate_n: int = 30, skip_answer: bool = False) -> Tuple[List[Dict], List[Dict], str]:
    """
    Run the optimized retrieval pipeline.
    Returns (selected_chunks_with_embeddings, raw_candidates, answer).
    """
    store = get_vector_store()
    query_emb = embed_query(query)

    # Get larger candidate pool WITH embeddings
    candidates = store.search_with_embeddings(query_emb, top_k=candidate_n)

    if not candidates:
        return [], [], ""

    # Run optimizer
    selected = optimize_selection(
        query_embedding=query_emb,
        candidates=candidates,
        k=top_k,
        alpha=settings.optimizer_alpha,
        beta=settings.optimizer_beta,
        gamma=settings.optimizer_gamma,
    )

    # We need embeddings for metrics — re-attach from candidates
    selected_with_embs = []
    candidate_map = {c["chunk_id"]: c for c in candidates}
    for s in selected:
        entry = s.copy()
        if s["chunk_id"] in candidate_map:
            entry["embedding"] = candidate_map[s["chunk_id"]]["embedding"]
        selected_with_embs.append(entry)

    if skip_answer:
        return selected_with_embs, candidates, ""

    # Generate answer
    from app.services.llm_chain import generate_answer

    chunks_for_llm = []
    for s in selected:
        c = s.copy()
        c.pop("embedding", None)
        chunks_for_llm.append(c)

    answer = generate_answer(query=query, chunks=chunks_for_llm)
    return selected_with_embs, candidates, answer


# ── Main Evaluation Loop ──────────────────────────────────


def evaluate(queries: List[Dict], skip_llm: bool = False, scores_only: bool = False, top_k: int = 10) -> Dict[str, Any]:
    """Run the full evaluation across all queries."""
    all_results = []

    # Aggregators
    topk_diversities = []
    opt_diversities = []
    redundancy_reductions = []
    faithfulness_topk_scores = []
    faithfulness_opt_scores = []
    reliability_topk_scores = []
    reliability_opt_scores = []

    for i, q in enumerate(queries):
        query = q["query"]
        logger.info("━" * 60)
        logger.info("Query %d/%d: %s", i + 1, len(queries), query[:80])

        skip_answer = scores_only or skip_llm

        # ── Run both pipelines ──
        try:
            topk_chunks, topk_answer = run_topk_pipeline(query, top_k=top_k, skip_answer=scores_only)
        except Exception as e:
            logger.error("Top-K pipeline failed: %s", e)
            continue

        # Small delay to avoid rate limits
        time.sleep(1)

        try:
            opt_chunks, _, opt_answer = run_optimized_pipeline(query, top_k=top_k, skip_answer=scores_only)
        except Exception as e:
            logger.error("Optimized pipeline failed: %s", e)
            continue

        if not topk_chunks or not opt_chunks:
            logger.warning("Skipping query — no chunks retrieved")
            continue

        # ── Embedding-Based Metrics ──
        topk_embs = _get_embeddings(topk_chunks)
        opt_embs = _get_embeddings([c for c in opt_chunks if "embedding" in c])

        topk_div = measure_diversity(topk_embs)
        opt_div = measure_diversity(opt_embs)
        redund_red = measure_redundancy_reduction(topk_embs, opt_embs)

        topk_diversities.append(topk_div)
        opt_diversities.append(opt_div)
        redundancy_reductions.append(redund_red)

        # ── Tri-Metric Scoring Components (Rel / Cov / Sup) ──
        query_emb_np = np.asarray(embed_query(query), dtype=np.float32)
        topk_scores = compute_scoring_components(query_emb_np, topk_embs)
        opt_scores = compute_scoring_components(query_emb_np, opt_embs)

        result_entry = {
            "query": query,
            "topk_diversity": round(topk_div, 1),
            "opt_diversity": round(opt_div, 1),
            "redundancy_reduction": round(redund_red, 1),
            "topk_near_dupes": count_near_duplicates(topk_embs),
            "opt_near_dupes": count_near_duplicates(opt_embs),
            "topk_rel": topk_scores["avg_rel"],
            "topk_cov": topk_scores["avg_cov"],
            "topk_sup": topk_scores["avg_sup"],
            "opt_rel": opt_scores["avg_rel"],
            "opt_cov": opt_scores["avg_cov"],
            "opt_sup": opt_scores["avg_sup"],
        }

        # ── LLM-Based Metrics ──
        if not skip_llm and not scores_only:
            time.sleep(2)  # Rate limit buffer

            # Build context strings for faithfulness check
            topk_context = "\n---\n".join(c["text"][:500] for c in topk_chunks)
            opt_context = "\n---\n".join(c["text"][:500] for c in opt_chunks)

            faith_topk = measure_faithfulness(topk_context, topk_answer)
            time.sleep(2)
            faith_opt = measure_faithfulness(opt_context, opt_answer)
            time.sleep(2)

            rel_topk = measure_reliability(query, topk_answer)
            time.sleep(2)
            rel_opt = measure_reliability(query, opt_answer)

            faithfulness_topk_scores.append(faith_topk)
            faithfulness_opt_scores.append(faith_opt)
            reliability_topk_scores.append(rel_topk)
            reliability_opt_scores.append(rel_opt)

            result_entry.update({
                "topk_faithfulness": faith_topk,
                "opt_faithfulness": faith_opt,
                "topk_reliability": rel_topk,
                "opt_reliability": rel_opt,
            })

        all_results.append(result_entry)
        logger.info(
            "  Diversity — TopK: %.1f%% | Opt: %.1f%% | Redundancy Reduction: %.1f%%",
            topk_div, opt_div, redund_red,
        )
        logger.info(
            "  Scores — TopK [R=%.4f C=%.4f S=%.4f] | Opt [R=%.4f C=%.4f S=%.4f]",
            topk_scores["avg_rel"], topk_scores["avg_cov"], topk_scores["avg_sup"],
            opt_scores["avg_rel"], opt_scores["avg_cov"], opt_scores["avg_sup"],
        )
        if not skip_llm and not scores_only:
            logger.info(
                "  Faithfulness — TopK: %.0f | Opt: %.0f | Reliability — TopK: %.0f | Opt: %.0f",
                faith_topk, faith_opt, rel_topk, rel_opt,
            )

        # Rate limit between queries
        time.sleep(2)

    # ── Aggregate Results ──
    summary = {
        "num_queries": len(all_results),
        "evidence_diversity_topk": round(float(np.mean(topk_diversities)), 1) if topk_diversities else 0,
        "evidence_diversity_opt": round(float(np.mean(opt_diversities)), 1) if opt_diversities else 0,
        "redundancy_reduction": round(float(np.mean(redundancy_reductions)), 1) if redundancy_reductions else 0,
    }

    if not skip_llm:
        summary.update({
            "hallucination_resistance_topk": round(float(np.mean(faithfulness_topk_scores)), 1) if faithfulness_topk_scores else 0,
            "hallucination_resistance_opt": round(float(np.mean(faithfulness_opt_scores)), 1) if faithfulness_opt_scores else 0,
            "answer_reliability_topk": round(float(np.mean(reliability_topk_scores)), 1) if reliability_topk_scores else 0,
            "answer_reliability_opt": round(float(np.mean(reliability_opt_scores)), 1) if reliability_opt_scores else 0,
        })

    return {"summary": summary, "per_query": all_results}


# ── Pretty Print ──────────────────────────────────────────


def print_results(results: Dict[str, Any]) -> None:
    """Print a formatted results table."""
    s = results["summary"]

    print("\n" + "=" * 70)
    print("  📊  EVALUATION RESULTS — Impact on Answer Quality")
    print("=" * 70)
    print(f"  Queries evaluated: {s['num_queries']}")
    print("-" * 70)
    print(f"  {'Metric':<35} {'Top-K':>10} {'Optimized':>10} {'Delta':>10}")
    print("-" * 70)

    topk_div = s["evidence_diversity_topk"]
    opt_div = s["evidence_diversity_opt"]
    print(f"  {'Evidence Diversity':<35} {topk_div:>9.1f}% {opt_div:>9.1f}% {opt_div - topk_div:>+9.1f}%")

    red = s["redundancy_reduction"]
    print(f"  {'Redundancy Reduction':<35} {'—':>10} {red:>9.1f}% {'':>10}")

    if "hallucination_resistance_opt" in s:
        faith_topk = s["hallucination_resistance_topk"]
        faith_opt = s["hallucination_resistance_opt"]
        print(f"  {'Hallucination Resistance':<35} {faith_topk:>9.1f}% {faith_opt:>9.1f}% {faith_opt - faith_topk:>+9.1f}%")

        rel_topk = s["answer_reliability_topk"]
        rel_opt = s["answer_reliability_opt"]
        print(f"  {'Answer Reliability':<35} {rel_topk:>9.1f}% {rel_opt:>9.1f}% {rel_opt - rel_topk:>+9.1f}%")

    print("=" * 70)

    # ── Per-Query Scoring Components Table ──
    per_query = results.get("per_query", [])
    if per_query and "topk_rel" in per_query[0]:
        print("\n" + "=" * 105)
        print("  📐  TRI-METRIC SCORING TABLE — Per-Query Rel / Cov / Sup")
        print("=" * 105)
        print(f"  {'Query':<40} {'Top-K':^30} {'Optimized':^30}")
        print(f"  {'':<40} {'Rel':>8} {'Cov':>10} {'Sup':>10} {'Rel':>8} {'Cov':>10} {'Sup':>10}")
        print("-" * 105)

        # Per query rows
        for pq in per_query:
            label = pq["query"][:38]
            print(
                f"  {label:<40}"
                f" {pq['topk_rel']:>8.4f} {pq['topk_cov']:>10.4f} {pq['topk_sup']:>10.4f}"
                f" {pq['opt_rel']:>8.4f} {pq['opt_cov']:>10.4f} {pq['opt_sup']:>10.4f}"
            )

        # Averages
        avg_topk_r = np.mean([p["topk_rel"] for p in per_query])
        avg_topk_c = np.mean([p["topk_cov"] for p in per_query])
        avg_topk_s = np.mean([p["topk_sup"] for p in per_query])
        avg_opt_r = np.mean([p["opt_rel"] for p in per_query])
        avg_opt_c = np.mean([p["opt_cov"] for p in per_query])
        avg_opt_s = np.mean([p["opt_sup"] for p in per_query])

        print("-" * 105)
        print(
            f"  {'AVERAGE':<40}"
            f" {avg_topk_r:>8.4f} {avg_topk_c:>10.4f} {avg_topk_s:>10.4f}"
            f" {avg_opt_r:>8.4f} {avg_opt_c:>10.4f} {avg_opt_s:>10.4f}"
        )
        print("=" * 105)

    # Print values for change.html update
    print("\n  📋 Values for change.html radar chart (opt_vals):")
    print(f"     Evidence Diversity:      {opt_div:.0f}")
    print(f"     Redundancy Reduction:    {red:.0f}")
    if "hallucination_resistance_opt" in s:
        print(f"     Hallucination Resistance: {s['hallucination_resistance_opt']:.0f}")
        print(f"     Answer Reliability:      {s['answer_reliability_opt']:.0f}")
    print()


# ── CLI Entry Point ───────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate ContextAware RAG quality metrics")
    parser.add_argument("--dry-run", action="store_true", help="Verify script loads without running evaluation")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based metrics (faster, no API cost for judge)")
    parser.add_argument("--scores-only", action="store_true", help="Only compute Rel/Cov/Sup scoring table (fastest, embedding API only)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to select (default: 10)")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries to evaluate")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run — verifying script loads OK")
        store = get_vector_store()
        logger.info("ChromaDB: %d chunks indexed", store.count)
        logger.info("Queries file: %s (%s)", QUERIES_PATH, "exists" if QUERIES_PATH.exists() else "MISSING")
        logger.info("Dry run complete ✓")
        return

    # Load queries
    if not QUERIES_PATH.exists():
        logger.error("Queries file not found: %s", QUERIES_PATH)
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if args.max_queries:
        queries = queries[:args.max_queries]

    mode_label = "scores-only" if args.scores_only else ("skip-llm" if args.skip_llm else "full")
    logger.info("Starting evaluation with %d queries (mode=%s, top_k=%d)", len(queries), mode_label, args.top_k)

    # Check store has data
    store = get_vector_store()
    chunk_count = store.count
    if chunk_count == 0:
        logger.error("ChromaDB is empty — please ingest documents first!")
        sys.exit(1)
    logger.info("ChromaDB: %d chunks indexed", chunk_count)

    # Run evaluation
    results = evaluate(queries, skip_llm=args.skip_llm, scores_only=args.scores_only, top_k=args.top_k)

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)

    # Print
    print_results(results)


if __name__ == "__main__":
    main()
