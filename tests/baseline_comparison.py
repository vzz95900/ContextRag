"""
Comprehensive Baseline Comparison — Top-K vs Hybrid vs MMR vs Multi-Objective.

Addresses reviewer comments:
  - Point 3: More queries (25), clear quantitative comparison with baselines
  - Point 4: Hallucination evidence via LLM-as-judge faithfulness scoring

Baselines implemented:
  1. Top-K:           Pure dense cosine similarity, top-k selection
  2. Hybrid (RRF):    Dense + BM25 with Reciprocal Rank Fusion
  3. MMR:             Maximal Marginal Relevance (lambda=0.5)
  4. Multi-Objective:  Proposed alpha*Rel + beta*Cov + gamma*Sup

Usage:
    cd d:\\Pjts\\CntextAware
    python tests/baseline_comparison.py                        # embedding metrics only (fast)
    python tests/baseline_comparison.py --with-faithfulness    # + LLM hallucination scoring
    python tests/baseline_comparison.py --dry-run              # verify script loads
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    optimize_selection,
)
from app.services.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).parent / "eval_queries_expanded.json"
OUTPUT_DIR = Path(__file__).parent / "ablation_results"
REDUNDANCY_THRESHOLD = 0.85


# =====================================================================
# BASELINE IMPLEMENTATIONS
# =====================================================================

def topk_selection(
    query_emb: List[float],
    candidates: List[Dict[str, Any]],
    cand_embs: List[np.ndarray],
    k: int,
) -> List[int]:
    """Baseline 1: Top-K by cosine similarity (standard dense retrieval)."""
    q = np.asarray(query_emb, dtype=np.float32)
    scores = [cosine_similarity(q, e) for e in cand_embs]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:k]


def hybrid_rrf_selection(
    query_text: str,
    query_emb: List[float],
    candidates: List[Dict[str, Any]],
    cand_embs: List[np.ndarray],
    k: int,
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> List[int]:
    """Baseline 2: Hybrid Dense + BM25 with Reciprocal Rank Fusion."""
    from rank_bm25 import BM25Okapi

    q = np.asarray(query_emb, dtype=np.float32)
    n = len(candidates)

    # Dense ranking
    dense_scores = [cosine_similarity(q, e) for e in cand_embs]
    dense_rank = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)

    # BM25 ranking
    tokenized = [candidates[i]["text"].lower().split() for i in range(n)]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query_text.lower().split())
    bm25_rank = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)

    # RRF fusion
    rrf_scores = [0.0] * n
    for rank, idx in enumerate(dense_rank):
        rrf_scores[idx] += alpha * (1.0 / (rank + rrf_k))
    for rank, idx in enumerate(bm25_rank):
        rrf_scores[idx] += (1.0 - alpha) * (1.0 / (rank + rrf_k))

    ranked = sorted(range(n), key=lambda i: rrf_scores[i], reverse=True)
    return ranked[:k]


def mmr_selection(
    query_emb: List[float],
    candidates: List[Dict[str, Any]],
    cand_embs: List[np.ndarray],
    k: int,
    lambda_param: float = 0.5,
) -> List[int]:
    """Baseline 3: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)."""
    q = np.asarray(query_emb, dtype=np.float32)
    n = len(candidates)

    # Pre-compute query-doc similarities
    query_sims = [cosine_similarity(q, e) for e in cand_embs]

    selected: List[int] = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        best_score = -float("inf")
        best_idx = -1

        for idx in remaining:
            relevance = query_sims[idx]

            # Max similarity to already selected
            if selected:
                max_sim = max(cosine_similarity(cand_embs[idx], cand_embs[s]) for s in selected)
            else:
                max_sim = 0.0

            # MMR = lambda * sim(q, d) - (1 - lambda) * max_sim(d, S)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def multiobj_selection(
    query_emb: List[float],
    candidates: List[Dict[str, Any]],
    k: int,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> List[int]:
    """Proposed: Multi-Objective Selection (alpha*Rel + beta*Cov + gamma*Sup)."""
    result = optimize_selection(
        query_embedding=query_emb,
        candidates=candidates,
        k=k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    # Return indices mapping back to candidate IDs
    cand_id_map = {c["chunk_id"]: i for i, c in enumerate(candidates)}
    return [cand_id_map[r["chunk_id"]] for r in result if r["chunk_id"] in cand_id_map]


# =====================================================================
# METRIC COMPUTATION
# =====================================================================

def compute_metrics(
    query_emb_np: np.ndarray,
    selected_embs: List[np.ndarray],
    topk_embs: List[np.ndarray],
) -> Dict[str, float]:
    """Compute all embedding-based metrics for a selection."""
    if not selected_embs:
        return {"rel": 0, "cov": 0, "sup": 0, "diversity": 0, "near_dupes": 0, "redund_red": 0}

    # Rel / Cov / Sup (simulating greedy build order)
    rels, covs, sups = [], [], []
    built: List[np.ndarray] = []
    for emb in selected_embs:
        rels.append(compute_relevance(query_emb_np, emb))
        covs.append(compute_coverage(emb, built))
        sups.append(compute_support(emb, built))
        built.append(emb)

    # Diversity
    if len(selected_embs) < 2:
        diversity = 100.0
    else:
        sims = []
        for i in range(len(selected_embs)):
            for j in range(i + 1, len(selected_embs)):
                sims.append(cosine_similarity(selected_embs[i], selected_embs[j]))
        diversity = (1.0 - float(np.mean(sims))) * 100

    # Near-duplicates
    near_dupes = 0
    for i in range(len(selected_embs)):
        for j in range(i + 1, len(selected_embs)):
            if cosine_similarity(selected_embs[i], selected_embs[j]) > REDUNDANCY_THRESHOLD:
                near_dupes += 1

    # Redundancy reduction vs top-k
    topk_dupes = 0
    for i in range(len(topk_embs)):
        for j in range(i + 1, len(topk_embs)):
            if cosine_similarity(topk_embs[i], topk_embs[j]) > REDUNDANCY_THRESHOLD:
                topk_dupes += 1

    if topk_dupes > 0:
        redund_red = ((topk_dupes - near_dupes) / topk_dupes) * 100
    else:
        redund_red = 100.0 if near_dupes == 0 else 0.0

    return {
        "rel": round(float(np.mean(rels)), 4),
        "cov": round(float(np.mean(covs)), 4),
        "sup": round(float(np.mean(sups)), 4),
        "diversity": round(diversity, 2),
        "near_dupes": near_dupes,
        "redund_red": round(redund_red, 1),
    }


# =====================================================================
# FAITHFULNESS (LLM-AS-JUDGE) — Point 4
# =====================================================================

FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given retrieved CONTEXT chunks and a generated ANSWER, score the answer's faithfulness to the context.

Faithfulness = what percentage of claims in the answer are directly supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0-100>, "reason": "<one sentence>"}}
"""


def measure_faithfulness(context_text: str, answer: str) -> float:
    """Hallucination resistance via LLM-as-judge faithfulness scoring."""
    from app.services.llm_chain import generate_answer

    prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    dummy_chunks = [{"text": "Evaluation task.", "metadata": {"filename": "eval", "page_num": 0}}]

    try:
        raw = generate_answer(query=prompt, chunks=dummy_chunks, history=None)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        result = json.loads(cleaned)
        return float(result.get("score", 50))
    except Exception as e:
        logger.warning("Faithfulness parse error: %s", e)
        return 50.0


def generate_answer_from_chunks(query: str, chunks: List[Dict]) -> str:
    """Generate an LLM answer from selected chunks."""
    from app.services.llm_chain import generate_answer

    clean_chunks = []
    for c in chunks:
        cc = c.copy()
        cc.pop("embedding", None)
        clean_chunks.append(cc)
    return generate_answer(query=query, chunks=clean_chunks)


# =====================================================================
# MAIN COMPARISON
# =====================================================================

BASELINES = [
    ("Top-K", "topk"),
    ("Hybrid (RRF)", "hybrid"),
    ("MMR (l=0.5)", "mmr"),
    ("Multi-Obj (Ours)", "multiobj"),
]


def run_comparison(
    queries: List[Dict],
    top_k: int = 10,
    candidate_n: int = 30,
    with_faithfulness: bool = False,
) -> Dict[str, Any]:
    """Run all baselines across all queries and collect metrics."""
    store = get_vector_store()
    logger.info("ChromaDB: %d chunks indexed", store.count)

    # Phase 1: Pre-compute embeddings
    logger.info("Phase 1: Embedding %d queries...", len(queries))
    query_data = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info("  [%d/%d] %s", i + 1, len(queries), query_text[:60])

        query_emb = embed_query(query_text)
        query_emb_np = np.asarray(query_emb, dtype=np.float32)

        candidates = store.search_with_embeddings(query_emb, top_k=candidate_n)
        if not candidates:
            logger.warning("  No candidates -- skipping")
            continue

        cand_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in candidates]

        # Top-k reference (for redundancy reduction computation)
        topk_ref = store.search_with_embeddings(query_emb, top_k=top_k)
        topk_ref_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in topk_ref]

        query_data.append({
            "query": query_text,
            "query_emb": query_emb,
            "query_emb_np": query_emb_np,
            "candidates": candidates,
            "cand_embs": cand_embs,
            "topk_ref_embs": topk_ref_embs,
        })
        time.sleep(0.5)

    logger.info("Phase 1 complete: %d queries ready\n", len(query_data))

    # Phase 2: Run baselines
    logger.info("Phase 2: Running %d baselines...", len(BASELINES))

    all_results = {}

    for label, method in BASELINES:
        logger.info("  Baseline: %s", label)

        per_query_metrics = []
        per_query_faithfulness = []

        for qd in query_data:
            # Select documents based on method
            if method == "topk":
                sel_indices = topk_selection(qd["query_emb"], qd["candidates"], qd["cand_embs"], top_k)
            elif method == "hybrid":
                sel_indices = hybrid_rrf_selection(qd["query"], qd["query_emb"], qd["candidates"], qd["cand_embs"], top_k)
            elif method == "mmr":
                sel_indices = mmr_selection(qd["query_emb"], qd["candidates"], qd["cand_embs"], top_k)
            elif method == "multiobj":
                sel_indices = multiobj_selection(qd["query_emb"], qd["candidates"], top_k)
            else:
                continue

            sel_embs = [qd["cand_embs"][i] for i in sel_indices]

            # Compute embedding metrics
            metrics = compute_metrics(qd["query_emb_np"], sel_embs, qd["topk_ref_embs"])
            per_query_metrics.append(metrics)

            # Faithfulness scoring (optional, costs LLM tokens)
            if with_faithfulness:
                sel_chunks = [qd["candidates"][i] for i in sel_indices]
                context_text = "\n---\n".join(c["text"][:500] for c in sel_chunks)

                try:
                    answer = generate_answer_from_chunks(qd["query"], sel_chunks)
                    time.sleep(2)
                    faith_score = measure_faithfulness(context_text, answer)
                    time.sleep(2)
                except Exception as e:
                    logger.warning("Faithfulness failed: %s", e)
                    faith_score = 50.0

                per_query_faithfulness.append(faith_score)

        # Aggregate
        result = {
            "label": label,
            "method": method,
            "num_queries": len(per_query_metrics),
            "avg_rel": round(float(np.mean([m["rel"] for m in per_query_metrics])), 4),
            "avg_cov": round(float(np.mean([m["cov"] for m in per_query_metrics])), 4),
            "avg_sup": round(float(np.mean([m["sup"] for m in per_query_metrics])), 4),
            "avg_diversity": round(float(np.mean([m["diversity"] for m in per_query_metrics])), 2),
            "avg_near_dupes": round(float(np.mean([m["near_dupes"] for m in per_query_metrics])), 1),
            "avg_redund_red": round(float(np.mean([m["redund_red"] for m in per_query_metrics])), 1),
        }

        if with_faithfulness and per_query_faithfulness:
            result["avg_faithfulness"] = round(float(np.mean(per_query_faithfulness)), 1)

        all_results[method] = result

        logger.info(
            "    -> Rel=%.4f  Cov=%.4f  Sup=%.4f  Div=%.1f%%  NearDup=%.1f  RedRed=%.1f%%",
            result["avg_rel"], result["avg_cov"], result["avg_sup"],
            result["avg_diversity"], result["avg_near_dupes"], result["avg_redund_red"],
        )
        if "avg_faithfulness" in result:
            logger.info("    -> Faithfulness=%.1f%%", result["avg_faithfulness"])

    return all_results


# =====================================================================
# OUTPUT: Console Table
# =====================================================================

def print_comparison_table(results: Dict[str, Dict], with_faithfulness: bool = False) -> None:
    """Print formatted baseline comparison table."""
    print("\n" + "=" * 120)
    print("  BASELINE COMPARISON -- Top-K vs Hybrid vs MMR vs Multi-Objective (N=%d queries)" % 
          next(iter(results.values()))["num_queries"])
    print("=" * 120)

    if with_faithfulness:
        header = f"  {'Method':<22} | {'Rel':>7} {'Cov':>7} {'Sup':>7} | {'Div%':>7} {'NDup':>6} {'RedR%':>7} | {'Faith%':>7}"
    else:
        header = f"  {'Method':<22} | {'Rel':>7} {'Cov':>7} {'Sup':>7} | {'Div%':>7} {'NDup':>6} {'RedR%':>7}"
    print(header)
    print("  " + "-" * 116)

    for method in ["topk", "hybrid", "mmr", "multiobj"]:
        if method not in results:
            continue
        r = results[method]
        line = (
            f"  {r['label']:<22} |"
            f" {r['avg_rel']:>7.4f} {r['avg_cov']:>7.4f} {r['avg_sup']:>7.4f} |"
            f" {r['avg_diversity']:>6.1f}% {r['avg_near_dupes']:>5.1f} {r['avg_redund_red']:>6.1f}%"
        )
        if with_faithfulness and "avg_faithfulness" in r:
            line += f" | {r['avg_faithfulness']:>6.1f}%"
        print(line)

    print("=" * 120)


# =====================================================================
# OUTPUT: LaTeX Table
# =====================================================================

def save_latex_comparison(results: Dict[str, Dict], path: Path, with_faithfulness: bool = False) -> None:
    """Generate LaTeX table for baseline comparison."""
    n_queries = next(iter(results.values()))["num_queries"]

    if with_faithfulness:
        col_spec = "l | c c c | c c c | c"
        headers = (
            r"\textbf{Method} & \textbf{Rel} & \textbf{Cov} & \textbf{Sup}"
            r" & \textbf{Div\%} & \textbf{NDup} & \textbf{RedR\%} & \textbf{Faith\%} \\"
        )
    else:
        col_spec = "l | c c c | c c c"
        headers = (
            r"\textbf{Method} & \textbf{Rel} & \textbf{Cov} & \textbf{Sup}"
            r" & \textbf{Div\%} & \textbf{NDup} & \textbf{RedR\%} \\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Baseline Comparison over {n_queries} Queries (k=10, N=30)}}",
        r"\label{tab:baseline_comparison}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        headers,
        r"\hline",
    ]

    for method in ["topk", "hybrid", "mmr", "multiobj"]:
        if method not in results:
            continue
        r = results[method]
        label = r["label"].replace("_", r"\_")

        row = (
            f"{label} & {r['avg_rel']:.4f} & {r['avg_cov']:.4f} & {r['avg_sup']:.4f}"
            f" & {r['avg_diversity']:.1f} & {r['avg_near_dupes']:.1f} & {r['avg_redund_red']:.1f}"
        )
        if with_faithfulness and "avg_faithfulness" in r:
            row += f" & {r['avg_faithfulness']:.1f}"

        # Bold our method
        if method == "multiobj":
            row = r"\textbf{" + row.replace(" & ", r"} & \textbf{") + "}"

        lines.append(row + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX saved: %s", path)


# =====================================================================
# OUTPUT: Matplotlib Chart
# =====================================================================

def save_comparison_chart(results: Dict[str, Dict], path: Path) -> None:
    """Generate grouped bar chart comparing baselines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping chart")
        return

    methods = ["topk", "hybrid", "mmr", "multiobj"]
    labels = [results[m]["label"] for m in methods if m in results]
    data = [results[m] for m in methods if m in results]

    rels = [d["avg_rel"] for d in data]
    covs = [d["avg_cov"] for d in data]
    sups = [d["avg_sup"] for d in data]
    divs = [d["avg_diversity"] / 100 for d in data]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5*width, rels, width, label="Relevance", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5*width, covs, width, label="Coverage", color="#55A868", edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5*width, sups, width, label="Support", color="#C44E52", edgecolor="white", linewidth=0.5)
    ax.bar(x + 1.5*width, divs, width, label="Diversity (/100)", color="#8172B2", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Retrieval Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Baseline Comparison: Top-K vs Hybrid vs MMR vs Multi-Objective", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight our method
    ax.axvspan(len(labels) - 1.45, len(labels) - 0.55, alpha=0.1, color="gold", zorder=0)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved: %s", path)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline comparison for research paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-faithfulness", action="store_true", help="Run LLM-as-judge faithfulness scoring")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-n", type=int, default=30)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    if args.dry_run:
        store = get_vector_store()
        logger.info("Dry run -- ChromaDB: %d chunks, Queries: %s",
                     store.count, "exists" if QUERIES_PATH.exists() else "MISSING")
        logger.info("Baselines: %s", [b[0] for b in BASELINES])
        logger.info("Dry run complete")
        return

    if not QUERIES_PATH.exists():
        logger.error("Queries file not found: %s", QUERIES_PATH)
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if args.max_queries:
        queries = queries[:args.max_queries]

    logger.info("Starting comparison: %d queries, %d baselines, top_k=%d, candidate_n=%d",
                len(queries), len(BASELINES), args.top_k, args.candidate_n)

    results = run_comparison(
        queries,
        top_k=args.top_k,
        candidate_n=args.candidate_n,
        with_faithfulness=args.with_faithfulness,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_comparison_table(results, with_faithfulness=args.with_faithfulness)

    # Save CSV
    csv_path = OUTPUT_DIR / "baseline_comparison.csv"
    fieldnames = ["label", "method", "num_queries", "avg_rel", "avg_cov", "avg_sup",
                  "avg_diversity", "avg_near_dupes", "avg_redund_red"]
    if args.with_faithfulness:
        fieldnames.append("avg_faithfulness")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in ["topk", "hybrid", "mmr", "multiobj"]:
            if m in results:
                writer.writerow(results[m])
    logger.info("CSV saved: %s", csv_path)

    # Save LaTeX
    latex_path = OUTPUT_DIR / "baseline_comparison_table.tex"
    save_latex_comparison(results, latex_path, with_faithfulness=args.with_faithfulness)

    # Save chart
    chart_path = OUTPUT_DIR / "baseline_comparison_chart.png"
    save_comparison_chart(results, chart_path)

    # Save JSON
    json_path = OUTPUT_DIR / "baseline_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("JSON saved: %s", json_path)

    print(f"\n  [OUTPUT] All saved to: {OUTPUT_DIR}")
    print(f"     - baseline_comparison.csv")
    print(f"     - baseline_comparison_table.tex")
    print(f"     - baseline_comparison_chart.png")
    print(f"     - baseline_comparison.json")
    print()


if __name__ == "__main__":
    main()
