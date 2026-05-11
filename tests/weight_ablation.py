"""
Weight Sensitivity Ablation — proves α, β, γ weights improve performance.

Runs the multi-objective optimizer with different weight configurations
across all evaluation queries and produces:
  1. Console summary table
  2. CSV file  (for external plotting)
  3. LaTeX table (copy-paste into paper)
  4. Matplotlib bar chart (saved as PNG)

Key optimisation: embeddings are computed ONCE per query, then reused
across all weight configs — so API cost = same as a single evaluation run.

Usage:
    cd d:\\Pjts\\CntextAware
    python tests/weight_ablation.py                 # full run (all configs)
    python tests/weight_ablation.py --dry-run       # verify script loads OK
    python tests/weight_ablation.py --top-k 5       # select 5 docs instead of 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Project root on sys.path ───────────────────────────────
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
OUTPUT_DIR = Path(__file__).parent / "ablation_results"

# Near-duplicate threshold (same as evaluate_quality.py)
REDUNDANCY_THRESHOLD = 0.85

# ── Weight Configurations to Test ──────────────────────────
# Each tuple: (label, alpha, beta, gamma)
WEIGHT_CONFIGS: List[Tuple[str, float, float, float]] = [
    ("Rel-Only",       1.0,  0.0,  0.0),   # pure relevance (≈ top-k baseline)
    ("Cov-Only",       0.0,  1.0,  0.0),   # pure coverage / diversity
    ("Sup-Only",       0.0,  0.0,  1.0),   # pure support / agreement
    ("Rel-Heavy",      0.7,  0.2,  0.1),   # relevance dominant
    ("Balanced",       0.5,  0.3,  0.2),   # default — paper's proposed weights
    ("Equal",          0.33, 0.33, 0.34),  # equal weighting
    ("Cov-Heavy",      0.2,  0.5,  0.3),   # coverage dominant
    ("Sup-Heavy",      0.2,  0.3,  0.5),   # support dominant
]


# ── Metric Helpers (reused from evaluate_quality.py) ───────

def measure_diversity(embeddings: List[np.ndarray]) -> float:
    """Evidence Diversity = (1 - mean pairwise cosine similarity) × 100."""
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


def compute_avg_scores(
    query_emb: np.ndarray,
    selected_embs: List[np.ndarray],
) -> Dict[str, float]:
    """Compute average Rel, Cov, Sup simulating greedy build order."""
    if not selected_embs:
        return {"avg_rel": 0.0, "avg_cov": 0.0, "avg_sup": 0.0}

    rels, covs, sups = [], [], []
    built: List[np.ndarray] = []

    for emb in selected_embs:
        rels.append(compute_relevance(query_emb, emb))
        covs.append(compute_coverage(emb, built))
        sups.append(compute_support(emb, built))
        built.append(emb)

    return {
        "avg_rel": round(float(np.mean(rels)), 4),
        "avg_cov": round(float(np.mean(covs)), 4),
        "avg_sup": round(float(np.mean(sups)), 4),
    }


# ── Core Ablation Logic ───────────────────────────────────

def run_ablation(
    queries: List[Dict],
    top_k: int = 10,
    candidate_n: int = 30,
) -> List[Dict[str, Any]]:
    """
    Run all weight configurations across all queries.

    Returns a list of result dicts, one per weight config, each containing
    averaged metrics across all queries.
    """
    store = get_vector_store()
    chunk_count = store.count
    logger.info("ChromaDB: %d chunks indexed", chunk_count)
    if chunk_count == 0:
        logger.error("ChromaDB is empty — ingest documents first!")
        sys.exit(1)

    # ── Phase 1: Pre-compute embeddings for all queries (ONE-TIME COST) ──
    logger.info("Phase 1: Embedding %d queries and retrieving candidate pools...", len(queries))
    query_data: List[Dict[str, Any]] = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info("  [%d/%d] %s", i + 1, len(queries), query_text[:60])

        query_emb = embed_query(query_text)
        query_emb_np = np.asarray(query_emb, dtype=np.float32)

        # Get candidate pool WITH embeddings
        candidates = store.search_with_embeddings(query_emb, top_k=candidate_n)
        if not candidates:
            logger.warning("  No candidates found — skipping")
            continue

        # Pre-convert to numpy
        cand_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in candidates]

        # Also get top-k baseline (pure cosine, no optimization)
        topk_results = store.search_with_embeddings(query_emb, top_k=top_k)
        topk_embs = [np.asarray(c["embedding"], dtype=np.float32) for c in topk_results]

        query_data.append({
            "query": query_text,
            "query_emb": query_emb,
            "query_emb_np": query_emb_np,
            "candidates": candidates,
            "cand_embs": cand_embs,
            "topk_embs": topk_embs,
        })

        # Rate-limit buffer for embedding API
        time.sleep(0.5)

    logger.info("Phase 1 complete: %d queries ready\n", len(query_data))

    # ── Phase 2: Run optimizer with each weight config (NO API CALLS) ──
    logger.info("Phase 2: Running %d weight configurations (CPU only)...", len(WEIGHT_CONFIGS))

    # Also compute top-k baseline metrics (once)
    topk_rels, topk_covs, topk_sups, topk_divs, topk_dupes = [], [], [], [], []
    for qd in query_data:
        scores = compute_avg_scores(qd["query_emb_np"], qd["topk_embs"])
        topk_rels.append(scores["avg_rel"])
        topk_covs.append(scores["avg_cov"])
        topk_sups.append(scores["avg_sup"])
        topk_divs.append(measure_diversity(qd["topk_embs"]))
        topk_dupes.append(count_near_duplicates(qd["topk_embs"]))

    topk_baseline = {
        "label": "Top-K (no opt)",
        "alpha": "-",
        "beta": "-",
        "gamma": "-",
        "avg_rel": round(float(np.mean(topk_rels)), 4),
        "avg_cov": round(float(np.mean(topk_covs)), 4),
        "avg_sup": round(float(np.mean(topk_sups)), 4),
        "diversity": round(float(np.mean(topk_divs)), 2),
        "near_dupes": round(float(np.mean(topk_dupes)), 1),
        "redund_reduction": "-",
    }

    all_config_results = [topk_baseline]

    for label, alpha, beta, gamma in WEIGHT_CONFIGS:
        logger.info("  Config: %s (α=%.2f, β=%.2f, γ=%.2f)", label, alpha, beta, gamma)

        per_query_rels = []
        per_query_covs = []
        per_query_sups = []
        per_query_divs = []
        per_query_dupes = []
        per_query_redund = []

        for qd in query_data:
            # Run optimizer with this weight config
            selected = optimize_selection(
                query_embedding=qd["query_emb"],
                candidates=qd["candidates"],
                k=top_k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            # Map selected chunk IDs back to their embeddings
            cand_map = {c["chunk_id"]: emb for c, emb in zip(qd["candidates"], qd["cand_embs"])}
            sel_embs = []
            for s in selected:
                if s["chunk_id"] in cand_map:
                    sel_embs.append(cand_map[s["chunk_id"]])

            # Compute metrics
            scores = compute_avg_scores(qd["query_emb_np"], sel_embs)
            per_query_rels.append(scores["avg_rel"])
            per_query_covs.append(scores["avg_cov"])
            per_query_sups.append(scores["avg_sup"])

            div = measure_diversity(sel_embs)
            per_query_divs.append(div)

            dupes = count_near_duplicates(sel_embs)
            per_query_dupes.append(dupes)

            # Redundancy reduction vs top-k
            topk_d = count_near_duplicates(qd["topk_embs"])
            if topk_d > 0:
                per_query_redund.append(((topk_d - dupes) / topk_d) * 100)
            else:
                per_query_redund.append(100.0 if dupes == 0 else 0.0)

        config_result = {
            "label": label,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "avg_rel": round(float(np.mean(per_query_rels)), 4),
            "avg_cov": round(float(np.mean(per_query_covs)), 4),
            "avg_sup": round(float(np.mean(per_query_sups)), 4),
            "diversity": round(float(np.mean(per_query_divs)), 2),
            "near_dupes": round(float(np.mean(per_query_dupes)), 1),
            "redund_reduction": round(float(np.mean(per_query_redund)), 1),
        }
        all_config_results.append(config_result)

        logger.info(
            "    → Rel=%.4f  Cov=%.4f  Sup=%.4f  Div=%.1f%%  RedundRed=%.1f%%",
            config_result["avg_rel"], config_result["avg_cov"],
            config_result["avg_sup"], config_result["diversity"],
            config_result["redund_reduction"],
        )

    return all_config_results


# ── Output: Console Table ─────────────────────────────────

def print_table(results: List[Dict]) -> None:
    """Print a formatted console table."""
    print("\n" + "=" * 120)
    print("  [RESULTS] WEIGHT SENSITIVITY ABLATION -- a, b, g Impact on Retrieval Quality")
    print("=" * 120)
    header = (
        f"  {'Configuration':<18} {'a':>5} {'b':>5} {'g':>5}"
        f" | {'Rel':>7} {'Cov':>7} {'Sup':>7}"
        f" | {'Diversity':>9} {'NearDup':>8} {'RedRed%':>8}"
    )
    print(header)
    print("  " + "-" * 116)

    for r in results:
        a = f"{r['alpha']:.2f}" if isinstance(r['alpha'], float) else str(r['alpha'])
        b = f"{r['beta']:.2f}" if isinstance(r['beta'], float) else str(r['beta'])
        g = f"{r['gamma']:.2f}" if isinstance(r['gamma'], float) else str(r['gamma'])
        rr = f"{r['redund_reduction']:.1f}" if isinstance(r['redund_reduction'], (int, float)) else str(r['redund_reduction'])

        print(
            f"  {r['label']:<18} {a:>5} {b:>5} {g:>5}"
            f" | {r['avg_rel']:>7.4f} {r['avg_cov']:>7.4f} {r['avg_sup']:>7.4f}"
            f" | {r['diversity']:>8.1f}% {r['near_dupes']:>7.1f} {rr:>7}%"
        )

    print("=" * 120)


# ── Output: CSV ───────────────────────────────────────────

def save_csv(results: List[Dict], path: Path) -> None:
    """Save results as CSV for external plotting."""
    fieldnames = ["label", "alpha", "beta", "gamma", "avg_rel", "avg_cov", "avg_sup",
                  "diversity", "near_dupes", "redund_reduction"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved: %s", path)


# ── Output: LaTeX Table ───────────────────────────────────

def save_latex(results: List[Dict], path: Path) -> None:
    """Generate a LaTeX table ready for copy-paste into the paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Weight Sensitivity Ablation — Impact of $\alpha$, $\beta$, $\gamma$ on Retrieval Quality}",
        r"\label{tab:weight_ablation}",
        r"\begin{tabular}{l c c c | c c c | c c}",
        r"\hline",
        r"\textbf{Config} & $\alpha$ & $\beta$ & $\gamma$ & \textbf{Rel} & \textbf{Cov} & \textbf{Sup} & \textbf{Diversity\%} & \textbf{RedRed\%} \\",
        r"\hline",
    ]

    for r in results:
        a = f"{r['alpha']:.2f}" if isinstance(r['alpha'], float) else "—"
        b = f"{r['beta']:.2f}" if isinstance(r['beta'], float) else "—"
        g = f"{r['gamma']:.2f}" if isinstance(r['gamma'], float) else "—"
        rr = f"{r['redund_reduction']:.1f}" if isinstance(r['redund_reduction'], (int, float)) else "—"

        label_escaped = r["label"].replace("_", r"\_")

        # Bold the "Balanced" row (proposed weights)
        if r["label"] == "Balanced":
            lines.append(
                rf"\textbf{{{label_escaped}}} & \textbf{{{a}}} & \textbf{{{b}}} & \textbf{{{g}}}"
                rf" & \textbf{{{r['avg_rel']:.4f}}} & \textbf{{{r['avg_cov']:.4f}}} & \textbf{{{r['avg_sup']:.4f}}}"
                rf" & \textbf{{{r['diversity']:.1f}}} & \textbf{{{rr}}} \\"
            )
        else:
            lines.append(
                rf"{label_escaped} & {a} & {b} & {g}"
                rf" & {r['avg_rel']:.4f} & {r['avg_cov']:.4f} & {r['avg_sup']:.4f}"
                rf" & {r['diversity']:.1f} & {rr} \\"
            )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX table saved: %s", path)


# ── Output: Matplotlib Chart ──────────────────────────────

def save_chart(results: List[Dict], path: Path) -> None:
    """Generate a grouped bar chart comparing weight configurations."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation")
        return

    # Skip top-k baseline for chart (it doesn't have α,β,γ)
    opt_results = [r for r in results if isinstance(r["alpha"], float)]

    labels = [r["label"] for r in opt_results]
    rels = [r["avg_rel"] for r in opt_results]
    covs = [r["avg_cov"] for r in opt_results]
    sups = [r["avg_sup"] for r in opt_results]
    divs = [r["diversity"] / 100 for r in opt_results]  # normalize to 0-1

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    bar1 = ax.bar(x - 1.5 * width, rels, width, label="Relevance", color="#4C72B0", edgecolor="white", linewidth=0.5)
    bar2 = ax.bar(x - 0.5 * width, covs, width, label="Coverage", color="#55A868", edgecolor="white", linewidth=0.5)
    bar3 = ax.bar(x + 0.5 * width, sups, width, label="Support", color="#C44E52", edgecolor="white", linewidth=0.5)
    bar4 = ax.bar(x + 1.5 * width, divs, width, label="Diversity (÷100)", color="#8172B2", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Weight Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Weight Sensitivity Ablation: Impact of α, β, γ on Retrieval Quality", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight the "Balanced" bar group
    balanced_idx = next((i for i, r in enumerate(opt_results) if r["label"] == "Balanced"), None)
    if balanced_idx is not None:
        ax.axvspan(balanced_idx - 0.45, balanced_idx + 0.45, alpha=0.1, color="gold", zorder=0)
        ax.annotate("Proposed", xy=(balanced_idx, 0.95), fontsize=9, ha="center",
                     fontstyle="italic", color="#B8860B")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved: %s", path)


# ── CLI Entry Point ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weight sensitivity ablation for research paper")
    parser.add_argument("--dry-run", action="store_true", help="Verify script loads OK without running")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to select (default: 10)")
    parser.add_argument("--candidate-n", type=int, default=30, help="Candidate pool size (default: 30)")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    if args.dry_run:
        store = get_vector_store()
        logger.info("Dry run — ChromaDB: %d chunks, Queries: %s",
                     store.count, "exists" if QUERIES_PATH.exists() else "MISSING")
        logger.info("Weight configs to test: %d", len(WEIGHT_CONFIGS))
        for label, a, b, g in WEIGHT_CONFIGS:
            logger.info("  %s: α=%.2f β=%.2f γ=%.2f", label, a, b, g)
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

    logger.info(
        "Starting weight ablation: %d queries × %d configs (top_k=%d, candidate_n=%d)",
        len(queries), len(WEIGHT_CONFIGS), args.top_k, args.candidate_n,
    )

    # Run ablation
    results = run_ablation(queries, top_k=args.top_k, candidate_n=args.candidate_n)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all outputs
    print_table(results)

    csv_path = OUTPUT_DIR / "weight_ablation.csv"
    save_csv(results, csv_path)

    latex_path = OUTPUT_DIR / "weight_ablation_table.tex"
    save_latex(results, latex_path)

    chart_path = OUTPUT_DIR / "weight_ablation_chart.png"
    save_chart(results, chart_path)

    # Save raw JSON too
    json_path = OUTPUT_DIR / "weight_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("JSON saved: %s", json_path)

    print(f"\n  [OUTPUT] All outputs saved to: {OUTPUT_DIR}")
    print(f"     - weight_ablation.csv       -- for spreadsheets/plotting")
    print(f"     - weight_ablation_table.tex  -- copy-paste into LaTeX paper")
    print(f"     - weight_ablation_chart.png  -- grouped bar chart figure")
    print(f"     - weight_ablation.json       -- raw data")
    print()


if __name__ == "__main__":
    main()
